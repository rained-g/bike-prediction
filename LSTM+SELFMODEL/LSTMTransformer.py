import torch
import torch.nn as nn


class LSTMTransformer(nn.Module):
    def __init__(self, input_size=13, hidden_size=256, num_heads=4, dropout=0.1, lstm_layers=2, out_feature=96):
        super(LSTMTransformer, self).__init__()
        # LSTM 层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=lstm_layers, batch_first=True)
        # Transformer 层
        # self.transformer = nn.Transformer(
        #     d_model=hidden_size,
        #     nhead=num_heads,
        #     num_encoder_layers=1,
        #     num_decoder_layers=1,
        #     dropout=dropout
        # )
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout),
            num_layers=1
        )
        self.fc = nn.Linear(hidden_size, out_feature)  # 最终的输出层

    def forward(self, x):
        # LSTM 处理
        lstm_out, (h_n, c_n) = self.lstm(x)
        # 调整维度以适应 Transformer 的输入
        # lstm_out = lstm_out[:, -1, :]
        # Transformer 处理
        # transformer_out = self.transformer(lstm_out, lstm_out)
        encoder_out = self.transformer_encoder(lstm_out)[:, -1, :]
        # 调整维度以适应最终的输出层
        # transformer_out = transformer_out.permute(1, 0, 2)
        output = self.fc(encoder_out)  # 取最后一个时间步的输出
        return output


if __name__ == '__main__':

    model = LSTMTransformer()

    sum = 0
    for i, module in model.named_parameters():
        print(i, module.numel())
        sum += module.numel()
    print(sum)