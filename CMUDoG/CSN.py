import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from NeuralNetwork import NeuralNetwork


class TransformerBlock(nn.Module):
    def __init__(self, input_size, is_layer_norm=False):
        super(TransformerBlock, self).__init__()
        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_morm = nn.LayerNorm(normalized_shape=input_size)

        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def FFN(self, X):
        return self.linear2(self.relu(self.linear1(X)))

    def forward(self, Q, K, V, episilon=1e-8):
        """
        :param Q: (batch_size, max_r_words, embedding_dim)
        :param K: (batch_size, max_u_words, embedding_dim)
        :param V: (batch_size, max_u_words, embedding_dim)
        :return: output: (batch_size, max_r_words, embedding_dim)  same size as Q
        """
        dk = torch.Tensor([max(1.0, Q.size(-1))]).cuda()
        Q_K = Q.bmm(K.permute(0, 2, 1)) / (torch.sqrt(dk) + episilon)
        # (batch_size, max_r_words, max_u_words)
        Q_K_score = F.softmax(Q_K, dim=-1)
        V_att = Q_K_score.bmm(V)
        if self.is_layer_norm:
            # (batch_size, max_r_words, embedding_dim)
            X = self.layer_morm(Q + V_att)
            output = self.layer_morm(self.FFN(X) + X)
        else:
            X = Q + V_att
            output = self.FFN(X) + X
        return output


class CSN(NeuralNetwork):
    def __init__(self, word_embeddings, args, logger=None):
        self.args = args
        super(CSN, self).__init__()

        self.word_embedding = nn.Embedding.from_pretrained(word_embeddings, freeze=True, padding_idx=0)

        self.alpha = 0.5
        self.gamma = args.gamma
        self.decay = args.decay
        self.selector_transformer = TransformerBlock(input_size=args.emb_size)
        self.W_word = nn.Parameter(data=torch.Tensor(args.gru_hidden * 2, args.gru_hidden * 2, 10))
        self.v = nn.Parameter(data=torch.Tensor(10, 1))
        self.linear_score = nn.Linear(in_features=4, out_features=1)

        self.A1 = nn.Parameter(data=torch.Tensor(args.gru_hidden * 2, args.gru_hidden * 2))
        self.A2 = nn.Parameter(data=torch.Tensor(args.gru_hidden * 2, args.gru_hidden * 2))
        self.A3 = nn.Parameter(data=torch.Tensor(args.gru_hidden * 2, args.gru_hidden * 2))

        self.transformer_utt = TransformerBlock(input_size=args.gru_hidden * 2)
        self.transformer_res = TransformerBlock(input_size=args.gru_hidden * 2)
        self.transformer_ur = TransformerBlock(input_size=args.gru_hidden * 2)
        self.transformer_ru = TransformerBlock(input_size=args.gru_hidden * 2)

        self.affine1 = nn.Linear(in_features=9 * 9 * 64, out_features=args.gru_hidden)
        self.affine2 = nn.Linear(in_features=9 * 9 * 64, out_features=args.gru_hidden)
        self.affine_attn = nn.Linear(in_features=args.gru_hidden, out_features=1)

        # self.affine1 = nn.Linear(in_features=4 * 4 * 64, out_features=args.gru_hidden)
        # self.affine2 = nn.Linear(in_features=4 * 4 * 64, out_features=args.gru_hidden)
        # self.affine_attn = nn.Linear(in_features=args.gru_hidden, out_features=1)

        self.cnn_2d_1 = nn.Conv2d(in_channels=6, out_channels=32, kernel_size=(3, 3))
        self.maxpooling1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.cnn_2d_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2))
        self.maxpooling2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.gru_sentence = nn.LSTM(input_size=args.emb_size, hidden_size=args.gru_hidden, batch_first=True, bidirectional=True)
        self.gru_acc1 = nn.LSTM(input_size=args.gru_hidden, hidden_size=args.gru_hidden, batch_first=True)
        self.gru_acc2 = nn.LSTM(input_size=args.gru_hidden, hidden_size=args.gru_hidden, batch_first=True)
        self.affine_out = nn.Linear(args.gru_hidden * 2, 1)

        self.level = args.level

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.2)
        self.init_weights()
        self.logger = logger
        self.logger.info(self)

    def init_weights(self):
        init.uniform_(self.W_word)
        init.uniform_(self.v)
        init.uniform_(self.linear_score.weight)

        init.xavier_normal_(self.A1)
        init.xavier_normal_(self.A2)
        init.xavier_normal_(self.A3)
        init.xavier_normal_(self.cnn_2d_1.weight)
        init.xavier_normal_(self.cnn_2d_2.weight)
        init.xavier_normal_(self.affine1.weight)
        init.xavier_normal_(self.affine2.weight)
        init.xavier_normal_(self.affine_attn.weight)
        init.xavier_normal_(self.affine_out.weight)
        for weights in [self.gru_acc1.weight_hh_l0, self.gru_acc1.weight_ih_l0]:
            init.orthogonal_(weights)
        for weights in [self.gru_acc2.weight_hh_l0, self.gru_acc2.weight_ih_l0]:
            init.orthogonal_(weights)
        for weights in [self.gru_sentence.weight_hh_l0, self.gru_sentence.weight_ih_l0]:
            init.orthogonal_(weights)

    def distance(self, A, B, C, epsilon=1e-6):
        M1 = torch.einsum("bud,dd,brd->bur", [A, B, C])
        A_norm = A.norm(dim=-1)
        C_norm = C.norm(dim=-1)
        M2 = torch.einsum("bud,brd->bur", [A, C]) / (torch.einsum("bu,br->bur", A_norm, C_norm) + epsilon)
        return M1, M2

    def word_level_selector(self, key, context):
        """
        :param key:  (bsz, num_utterances, max_u_words, d)
        :param context:  (bsz, num_documents, max_d_words, d)
        :return: score:
        """
        dk = torch.sqrt(torch.Tensor([key.size(-1)])).cuda()
        A = torch.tanh(torch.einsum("blrd,ddh,bmud->blmruh", context, self.W_word, key) / dk)
        A = torch.einsum("blmruh,hp->blmrup", A, self.v).squeeze()   # (bsz, num_documents, num_utterances, max_d_words, max_u_words)
        s1 = A.max(dim=4)[0]  # (bsz, num_document, num_utterances, max_d_words)
        # s1 = torch.softmax(self.linear_word(a).squeeze(), dim=-1)  # b x l
        return s1

    def utterance_level_selector(self, key, context):
        """
        :param key:  (bsz, num_utterances, max_u_words, d)
        :param context:  (bsz, num_documents, max_d_words, d)
        :return: score:
        """
        key = key.mean(dim=2)
        context = context.mean(dim=2)
        norm_term = torch.einsum("bd,bu->bdu", torch.norm(context, dim=-1), torch.norm(key, dim=-1))
        s2 = torch.einsum("bdh,buh->bdu", context, key) / (1e-6 + norm_term)
        return s2

    def persona_selector(self, context, personas, level="word"):
        sp1, sp2, sp3, sp4 = personas.size()
        sc1, sc2, sc3, sc4 = context.size()
        if level == "word":
            multi_match_score = self.word_level_selector(context, personas)  # [batch, num_document, num_utterances, max_d_words]
            multi_match_score = multi_match_score.permute(0, 1, 3, 2)
            decay_factor = torch.ones((sc1, sp2, sp3, sc2)).cuda()
            for i in range(sc2):
                decay_factor[:, :, :, - i - 1] = self.decay ** i
            multi_match_score = multi_match_score * decay_factor
            # match_score = multi_match_score.mean(dim=-1)
            match_score = self.linear_score(multi_match_score).squeeze(dim=-1)  # (batch_size, num_personas, max_p_words)
            mask = (match_score.sigmoid() >= self.gamma).float()
            match_score = match_score * mask  # (batch_size, num_personas, max_p_words)
            personas = personas * match_score.unsqueeze(dim=-1)
        else:
            multi_match_score = self.utterance_level_selector(context, personas)  # [batch, num_persona, num_utterance]
            decay_factor = torch.ones((sc1, sp2, sc2)).cuda()
            for i in range(sc2):
                decay_factor[:, :, -i - 1] = self.decay ** i
            # match_score = multi_match_score.mean(dim=-1)
            multi_match_score = multi_match_score * decay_factor
            match_score = self.linear_score(multi_match_score).squeeze(dim=-1)  # (batch_size, num_personas)
            mask = (match_score.sigmoid() >= self.gamma).float()
            match_score = match_score * mask  # (batch_size, num_personas)
            personas = personas * match_score.unsqueeze(dim=-1).unsqueeze(dim=-1)
        return personas

    def get_Matching_Map(self, bU_embedding, bR_embedding):
        """
        :param bU_embedding: (batch_size*max_utterances, max_u_words, embedding_dim)
        :param bR_embedding: (batch_size*max_utterances, max_r_words, embedding_dim)
        :return: E: (bsz*max_utterances, max_u_words, max_r_words)
        """
        M1, M2 = self.distance(bU_embedding, self.A1, bR_embedding)
        Hu = self.transformer_utt(bU_embedding, bU_embedding, bU_embedding)
        Hr = self.transformer_res(bR_embedding, bR_embedding, bR_embedding)
        M3, M4 = self.distance(Hu, self.A2, Hr)
        Hur = self.transformer_ur(bU_embedding, bR_embedding, bR_embedding)
        Hru = self.transformer_ru(bR_embedding, bU_embedding, bU_embedding)
        M5, M6 = self.distance(Hur, self.A3, Hru)
        M = torch.stack([M1, M2, M3, M4, M5, M6], dim=1)  # (bsz*max_utterances, channel, max_u_words, max_r_words)
        return M

    def UR_Matching(self, bU_embedding, bR_embedding, type_m):
        """
        :param bU_embedding: (batch_size*max_utterances, max_u_words, embedding_dim)
        :param bR_embedding: (batch_size*max_utterances, max_r_words, embedding_dim)
        :return: (bsz*max_utterances, (max_u_words - width)/stride + 1, (max_r_words -height)/stride + 1, channel)
        """
        M1, M2 = self.distance(bU_embedding, self.A1, bR_embedding)
        Hu = self.transformer_utt(bU_embedding, bU_embedding, bU_embedding)
        Hr = self.transformer_res(bR_embedding, bR_embedding, bR_embedding)
        M3, M4 = self.distance(Hu, self.A2, Hr)
        Hur = self.transformer_ur(bU_embedding, bR_embedding, bR_embedding)
        Hru = self.transformer_ru(bR_embedding, bU_embedding, bU_embedding)
        M5, M6 = self.distance(Hur, self.A3, Hru)
        M = torch.stack([M1, M2, M3, M4, M5, M6], dim=1)
        Z = self.relu(self.cnn_2d_1(M))
        Z = self.maxpooling1(Z)
        Z = self.relu(self.cnn_2d_2(Z))
        Z = self.maxpooling2(Z)
        Z = Z.view(Z.size(0), -1)  # (bsz*max_utterances, *)
        if type_m == 1:
            V = self.tanh(self.affine1(Z))   # (bsz*max_utterances, 300)
        else:
            V = self.tanh(self.affine2(Z))   # (bsz*max_utterances, 300)
        return V

    def forward(self, bU, bR, bP, is_test=False):
        """
        :param bU: batch utterance, size: (batch_size, max_utterances, max_u_words)
        :param bR: batch responses, size: (batch_size, max_r_words)
        :param bP: batch profiles, size: (batch_size, max_num_profiels, max_p_words)
        :return: scores, size: (batch_size, )
        """
        bU_embedding = self.dropout(self.word_embedding(bU))
        bR_embedding = self.dropout(self.word_embedding(bR))
        bP_embedding = self.dropout(self.word_embedding(bP))

        su1, su2, su3, su4 = bU_embedding.size()  # (batch_size * max_utterances, max_u_words, embedding_dim)
        sr1, sr2, sr3 = bR_embedding.size()  # (batch_size, max_r_words, embedding_dim)
        sp1, sp2, sp3, sp4 = bP_embedding.size()

        bU_embedding = bU_embedding.view(su1 * su2, su3, su4)
        bP_embedding = bP_embedding.view(sp1 * sp2, sp3, sp4)
        bU_embedding, _ = self.gru_sentence(bU_embedding)
        bR_embedding, _ = self.gru_sentence(bR_embedding)
        bP_embedding, _ = self.gru_sentence(bP_embedding)

        bU_embedding = bU_embedding.view(su1, su2, su3, -1)  # (batch_size, max_utterances, max_u_words, embedding_dim)
        bP_embedding = bP_embedding.view(sp1, sp2, sp3, -1)

        bP_embedding = self.persona_selector(bU_embedding, bP_embedding, level=self.level)

        bR = bR_embedding.unsqueeze(dim=1).repeat(1, su2, 1, 1)
        bU = bU_embedding.view(-1, su3, self.args.gru_hidden * 2)
        bR = bR.view(-1, sr2, self.args.gru_hidden * 2)
        V1 = self.UR_Matching(bU, bR, 1)
        V1 = V1.view(su1, su2, -1)  # (bsz, max_utterances, 300)
        H1, _ = self.gru_acc1(V1)  # (bsz, max_utterances, rnn_hidden)
        L1 = self.dropout(H1[:, -1, :])  # (bsz, rnn_hidden)

        bR = bR_embedding.unsqueeze(dim=1).repeat(1, sp2, 1, 1)
        bP = bP_embedding.view(-1, sp3, self.args.gru_hidden * 2)
        bR = bR.view(-1, sr2, self.args.gru_hidden * 2)
        V2 = self.UR_Matching(bP, bR, 2)
        V2 = V2.view(sp1, sp2, -1)  # (bsz, max_personas, 300)
        H2, _ = self.gru_acc2(V2)  # (bsz * num_candidates, max_utterances, rnn_hidden)
        L2 = self.dropout(H2[:, -1, :])

        output = torch.sigmoid(self.affine_out(torch.cat([L1, L2], dim=-1))).squeeze(-1)
        return output
