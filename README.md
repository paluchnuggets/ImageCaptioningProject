# Image Captioning Project - Tell me what do you see

In this repository you can find Image Captioning model which I've trained as a part of the Udacity Computer Vision Nanodegree. Image captioning is the process when "seeing" an image the model is able to generate the sequence of words describing situation of that spesific image. See example below.<br>

CNN Encoder providing input to RNN Decoder.<br>
<img src="images/encoder-decoder.png"><br>
Credits: Udacity Computer vision Nanodegree

Above, the high-level description of that process can be seen. First the image is processed through CNN network that produces feature vector for that particular image. Next, that vector goes through the embedding layer which adjust its size to that required by the RNN. When trained, the RNN takes that embedded iamge vector and based on that produces the post probable sequence of words that describes it (based on weights matrix obtain during the training process).<br>

Whole training process is shown in this [notebook](/Training.ipynb) :running: and the model capabilities are shown [here](/Inference.ipynb) :muscle:.
