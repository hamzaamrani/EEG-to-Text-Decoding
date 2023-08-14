import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from torch.nn.utils.rnn import pack_padded_sequence

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=1024,
        dropout=0.1
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        return x
    
class BrainTranslator(nn.Module):
    def __init__(self, bart, in_feature = 840, decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048):
        super(BrainTranslator, self).__init__()
        
        # Embedded EEG raw features
        self.hidden_dim=512
        self.feature_embedded = FeatureEmbedded(input_dim=104, hidden_dim=self.hidden_dim)
        self.fc = ProjectionHead(embedding_dim=in_feature,projection_dim=in_feature,dropout=0.1) #nn.Linear(in_feature, in_feature)

        # conv1d
        self.conv1d_point = nn.Conv1d(1, 64, 1, stride=1)

        SUBJECTS = ['ZAB', 'ZDM', 'ZDN', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW', 'ZMG', 'ZPH', 
            'YSD', 'YFS', 'YMD', 'YAC', 'YFR', 'YHS', 'YLS', 'YDG', 'YRH', 'YRK', 'YMS', 'YIS', 'YTL', 'YSL', 'YRP', 'YAG', 'YDR', 'YAK']
        self.subjects_map_id = {}
        for i in range(len(SUBJECTS)):
            self.subjects_map_id[SUBJECTS[i]]=i
        
        # learnable subject matrices
        self.subject_matrices = []
        for i in range(len(SUBJECTS)):
            self.subject_matrices.append( nn.Parameter(torch.randn(64, 1)) )
        
        # Brain transformer encoder
        self.pos_embedding = nn.Parameter(torch.randn(1, 56, in_feature))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=in_feature, nhead=additional_encoder_nhead,  dim_feedforward = additional_encoder_dim_feedforward, dropout=0.1, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=12)
        self.layernorm_embedding = nn.LayerNorm(in_feature, eps=1e-05)

        self.brain_projection = ProjectionHead(embedding_dim=in_feature,projection_dim=1024,dropout=0.2)
        
        # BART
        self.bart = bart
        
    def freeze_pretrained_bart(self):
        for name, param in self.named_parameters():
            param.requires_grad = True
            if ('bart' in name):
                param.requires_grad = False

    def freeze_pretrained_brain(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
            if ('bart' in name):
                param.requires_grad = True

    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted, lenghts_words, word_contents_batch, word_contents_attn_batch, stepone, subject_batch, device, features=False):
        feature_embedding = self.feature_embedded(input_embeddings_batch, lenghts_words, device)
        if len(feature_embedding.shape)==2:
            feature_embedding = torch.unsqueeze(feature_embedding,0)
        encoded_embedding = self.fc(feature_embedding)

        # subject layer
        encoded_embedding_subject=[]
        for i in range(encoded_embedding.shape[0]):
            tmp = torch.unsqueeze(encoded_embedding[i,:,:],1)
            tmp = self.conv1d_point(tmp)
            tmp = torch.swapaxes(tmp, 1, 2)
            mat_subject = self.subject_matrices[self.subjects_map_id[subject_batch[i]]].to(device)
            tmp = torch.matmul(tmp, mat_subject)
            tmp = torch.squeeze(tmp)
            encoded_embedding_subject.append(tmp)        
        if len(encoded_embedding_subject) == 1:
            encoded_embedding_subject = torch.unsqueeze(encoded_embedding_subject[0],0)
        else:
            encoded_embedding_subject = torch.stack(encoded_embedding_subject,0).to(device)

        brain_embedding = encoded_embedding_subject + self.pos_embedding
        brain_embedding = self.encoder(brain_embedding, src_key_padding_mask = input_masks_invert)
        brain_embedding = self.layernorm_embedding(brain_embedding)
        
        brain_embedding = self.brain_projection(brain_embedding)

        if stepone==True:
            words_embedding = self.bart.model.encoder.embed_tokens(word_contents_batch)
            loss = nn.MSELoss()
            return loss(brain_embedding, words_embedding)
        else:
            out = self.bart(inputs_embeds = brain_embedding, attention_mask = input_masks_batch, return_dict = True,  labels = target_ids_batch_converted)
            if features==True:
                return out.logits, brain_embedding
                
            return out.logits


class FeatureEmbedded(nn.Module):
    def __init__(self, input_dim=105, hidden_dim=512, num_layers=2, is_bidirectional=True):
        super(FeatureEmbedded, self).__init__()

        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.is_bidirectional=is_bidirectional
        
        self.lstm = nn.GRU(input_size=self.input_dim, 
                       hidden_size=self.hidden_dim, 
                       num_layers=self.num_layers, 
                       batch_first=True, 
                       dropout=0.2,
                       bidirectional=self.is_bidirectional
        )
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)
                
    def forward(self, x, lenghts, device):

        sentence_embedding_batch=[]
        for x_sentence, lenghts_sentence in zip(x,lenghts):

            lstm_input = pack_padded_sequence(x_sentence, lenghts_sentence.cpu().numpy(), batch_first=True, enforce_sorted=False)
            lstm_outs, hidden  = self.lstm(lstm_input)
            lstm_outs, _ = nn.utils.rnn.pad_packed_sequence(lstm_outs)  

            if not self.is_bidirectional:
                sentence_embedding = []
                for i in range(lenghts_sentence.shape[0]):
                    sentence_embedding.append(lstm_outs[int(lenghts_sentence[i]-1),i,:])
                sentence_embedding=torch.stack(sentence_embedding,0)  #lstm_outs[-1]
            else:
                sentence_embedding = []
                for i in range(lenghts_sentence.shape[0]):
                    sentence_embedding.append(lstm_outs[int(lenghts_sentence[i]-1),i,:])
                sentence_embedding=torch.stack(sentence_embedding,0)

            sentence_embedding_batch.append(sentence_embedding)

        return torch.squeeze(torch.stack(sentence_embedding_batch,0)).to(device)