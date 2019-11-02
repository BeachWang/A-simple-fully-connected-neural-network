# README
<p> baseline：基础的3层全连接神经网络，两个隐藏层分别有300和100个神经元，学习率为0.02，采用Mini-Batch GD，每组batch有64张图，网络中训练时对每组batch进行进行normalization，激活函数采用ReLU。<p>
<p> (1) layers：在baseline的基础上分别加一个隐藏层和减少一个隐藏层的结果。一个隐藏层时，隐藏层神经元个数为250。三个隐藏层时神经元个数分别为450、250和100。<p>
<p> (2) BGD&SGD：在baseline的基础上分别将训练方法改成BGD和SGD的结果。<p>
<p> (3) initializtions：在baseline的基础上分别用xavier和kaiming进行初始化。<p>
<p> (4) learningrate：在baseline的基础上分别运用StepLR和余弦退火调整学习率 CosineAnnealingLR学习率优化算法。 <p>
<p> (5) rgularizations: 在baseline的基础上分别对各层上的参数求第一范式和第二范式加权到loss上。 <p>
