# minic-learning
一般我们都将神经网络放置到服务器上进行训练，所以在搭建和训练神经网络的时候并没有考虑到计算器资源的大小等等。但是当我们需将模型落地到某一硬件设施之中，如机器人或手机，设备的存储容量以及算力就成为模型的限制因素。为了使得模型能够嵌入到某一设备之中，我们通常会将模型进行压缩。压缩的方法主要有以下几种：  
![image](image/network_compression.png)  
在该项目中，我采用了第二种方法--知识蒸馏，来压缩网络模型的大小。在这里，我首先使用cifar10数据集来训练一个较大的网络。之后，我再使用teacher-student的模式来训练一个较小的网络。teacher-student模式模仿了老师与学生的关系，将teacher model所学到的知识教授给student model。在teacher-student训练模型中，模型的训练空间不再是label space，而是logits space或者是probability space。据研究，模型在logits space或probability space更加容易训练。而且在logits space或probability space上，student model不再只学习到单一的标签信息，而会学习到多样的有效信息 。如teacher model给出的概率分布不仅会告诉student model输入的label还是告诉其input与其它label对应input的相似度，丰富了student model的知识。
# Environment
python 3.7  
pytorch 1.5  
torchvision 0.6  
opencv 3.4
# Experiment
实验分为三组：训练vgg11网络，训练vgg16网络和使用vgg16作为teacher model来训练vgg11网络。为了提高vgg16的泛化能力，使得vgg16和vgg11两者之间有明显的差别，我使用了pytorch中vgg16的预训练模型来初始化vgg16的参数。三组实验的实验数据分别如下所示，在训练过程中，三者使用相同的超参数设定。  
![image](image/vgg11.jpg)  
![image](image/vgg16.jpg)  
![image](image/teacher_student.jpg)  
从上述的实验可知，与一般方法训练的vgg11相比，使用teacher-student模式训练的vgg11在精度上提高了2%左右。但是由于vgg11模型在cifar10数据集上本身就能够work，因为其train loss处于下降的趋势。所以，该实验并无法验证teacher-student的训练方法能够降低模型的训练难度。所以，接下来我想或许我们能够使用一个更加浅的网络来充当student网络来验证teacher-student的实用性。  
由于student是在logits或probability空间上计算，所以它也可以使用mse来作为loss function。而且mse很有可能使得模型更加收敛或取得更好的性能。这只是个人直觉。模型的学习曲线和准确率如下所示，除了降低原先的学习率之外，其它的别无改动。  
![image](image/teacher_student_mse.jpg)  
从图中可知，模型的准确率达到了77%，其超越了使用信息熵为loss function训练的模型，同时其性能逼近teacher model。那么，为什么mse能够使得模型的性能提高呢？我绘画了一元的信息熵和均方差函数，如下所示:  
![image](image/crossentropy.jpg)  
![image](image/mse.jpg)  

在crossentropy曲线中，最小值点的右边梯度较低，趋于0。这会导致模型的其区域内很难得到更新。而在最小值点左边的梯度较大，它会有可能使得模型掉入最小值点的左边或更远处。这使得模型训练不稳定以及难以到达最小值点。反观mse，其便没有如此缺点。这给我的启示是选择合适的loss function对模型的训练很重要。
