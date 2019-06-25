# coupletpy
## 对对联

## 数据的处理，在input层，数据加上 \</s> 结束符
## 在output层，数据开始加上 \<s> 开始符 结尾加上 \</s>结束符
## 序列长度计算的是序列本身的长度加一

## 对输入是序列本身长度加上\</s>结束符，即输入到attention_mechinism 里面的长度
## 对输出，训练时是 \<s> 加上序列本身长度，即train_helper 里面的sequence_length
## 计算loss 时需要进行mask，此时需要拿预测输出与原有的标签比较 比较的序列是原序列本身加上 \</s> 即 tf.sequence_mask 里面传入 y_input_len


## 采用传统attention
## encoder 层采用 双向lstm
## decoder 层采用 单向lstm

## decoder 层的num_units 和 encoder 层的num_units 要相等 decoder 的layersize 是encoder 层的二倍
## 这样是为了保证能够全部用到 encoder 层最后输出的 encoder_state

## 训练时采用 train_helper 预测时采用 greedyEmbeddingHelper

## loss 采用的是sequence loss 这是和自己手写一样的。
## 最后的train_op 采用的是 cliped gradients 这样是为了防止梯度爆炸，使得最大梯度更新不超过-1，1 
