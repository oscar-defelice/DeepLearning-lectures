# Tensorflow.js Examples

Here we show how to train and then deploy a model in a webapp.
We build a simple html page where we serve our model thanks to a JavaScript code.

## Installation
The key package to install is `tensorflowjs`. In a virtual env, one can install it simply by,
```bash
pip install tensorflowjs
```

## Examples

### Cat-vs-Dog classifier
The first example we build is a classic in computer vision: **cat-vs-dog classifier**.
1. [We use a notebook to train the model](https://github.com/oscar-defelice/DeepLearning-lectures/blob/master/TFJS/Example-CatVsDog/TFJS-trainModel.ipynb)
2. [We build a webapp to deploy our model](https://github.com/oscar-defelice/DeepLearning-lectures/blob/master/TFJS/Example-CatVsDog/index.html).

### Rock-Paper-Scissors game
Another example is the famous Rock-Paper-Scissor game, that is in principle very similar, but now we fine-tune the model directly in JavaScript and we train with live images from webcam.

![image](https://user-images.githubusercontent.com/49638680/115159438-60d2da80-a093-11eb-806d-7f0c2374a74f.png)

### Bonus I
I put here another example recognising hand gestures [here](https://github.com/oscar-defelice/handgesture.github.io),
while the relative code can be found at this [repo link](https://oscar-defelice.github.io/handgesture.github.io/).

<a href="https://oscar-defelice.github.io/handgesture.github.io/" rel="handgesture recognition" target="_blank">![hand](https://user-images.githubusercontent.com/49638680/114884954-7b445400-9e06-11eb-89d2-fe0c92962781.png)</a>

### Bonus II

You can play PacMan with your webcam!

![image](https://user-images.githubusercontent.com/49638680/115876770-02b54700-a447-11eb-869b-6a0ce8978585.png)

### [🕹️ Enjoy!](https://storage.googleapis.com/tfjs-examples/webcam-transfer-learning/dist/index.html)
