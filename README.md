# Deep-Dream

DeepDream is an experiment that visualizes the patterns learned by a neural network. Similar to when a child watches clouds and tries to interpret random shapes, DeepDream over-interprets and enhances the patterns it sees in an image.

The idea in DeepDream is to choose a layer (or layers) and maximize the "loss" in a way that the image increasingly "excites" the layers. The complexity of the features incorporated depends on layers chosen by you, i.e, lower layers produce strokes or simple patterns, while deeper layers give sophisticated features in images, or even whole objects.

The loss is the sum of the activations in the chosen layers. The loss is normalized at each layer so the contribution from larger layers does not outweigh smaller layers. Normally, loss is a quantity you wish to minimize via gradient descent. In DeepDream, you will maximize this loss via gradient ascent.

# Brain

| | | 
|:-------------------------:|:-------------------------:|
|<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="./did_you_wash_your_hand/brain2.png"> |  <img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="./did_you_wash_your_hand/brain5.png">| 
|<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="./did_you_wash_your_hand/brain7.png">   |  <img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="./did_you_wash_your_hand/brain6.png"> | 


| | | 
|:-------------------------:|:-------------------------:|
|<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="./did_you_wash_your_hand/brain2_conv2d_1380activation_1399.png"> |  <img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="./did_you_wash_your_hand/brain5_activation_580.png">| 
|<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="./did_you_wash_your_hand/brain7_conv2d_426.png">   |  <img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="./did_you_wash_your_hand/brain6_conv2d_37.png"> | 

# Deep Learning

| | | 
|:-------------------------:|:-------------------------:|
|<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="./did_you_wash_your_hand/brain2.png"> |  <img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="./did_you_wash_your_hand/brain5.png">| 
|<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="./did_you_wash_your_hand/brain7.png">   |  <img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="./did_you_wash_your_hand/brain6.png"> | 



