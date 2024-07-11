# ViT-vs-CNN-MNIST
Comparing the performance of Vision Transformer vs CNN (VGG11) on MNIST dataset.

## MNIST
Tackling the MNIST Handwriting Digit recognition problem.

## CNN
Builds the VGG11 CNN using PyTorch with evaluation of data augmentation (random flips, noise).

Results:

<img width="717" alt="Screenshot 2024-07-11 at 4 00 11 PM" src="https://github.com/shuh2002/ViT-vs-CNN-MNIST/assets/40676497/190a9f8d-5f33-4b35-9ddc-137aab01eedd">

- We can see that flipping our test images either horizontal or vertical reduces the accuracy to around 40 percent. Further, additional epochs of training don’t seem to increase the accuracy.
On deeper investigation, we can note that for horizontally-flipped images, our model predicts digits 0, 1, 4, and 8 with high accuracy and 5, 2, 7 with low accuracy (often mistaking one for the other). Similarly, for vertically-flipped images, our model predicts digits 3, 8, 1, 0, and 4 with high accuracy and 7, 2, 5, and 9 with low accuracy.
This tracts given the horizontal and vertical symmetry of each digit.
<img width="456" alt="Screenshot 2024-07-11 at 4 00 42 PM" src="https://github.com/shuh2002/ViT-vs-CNN-MNIST/assets/40676497/571ed321-05db-4d86-8ecf-b6a751d4809f">

- We can see that there is practically no perceptible difference in adding noise with variance 0.1 or 0.01. However, we see a significant dip in accuracy when we add noise with variance 1, implying that our model is not very resistant to added noise.
<img width="456" alt="Screenshot 2024-07-11 at 4 01 11 PM" src="https://github.com/shuh2002/ViT-vs-CNN-MNIST/assets/40676497/34583ebf-b5e0-4c66-bc94-1672803e4b96">

- Lastly, we augment the data by adding noise (variance = 2) to all training images, then flip 30% of them horizontal or vertical respectively. Doing so, we see that our accuracy for random flips and noise improves significantly, even if training takes longer and is overall less accurate.
Note that we also augment our test set (no noise) with similar flippings. 
<img width="721" alt="Screenshot 2024-07-11 at 4 01 40 PM" src="https://github.com/shuh2002/ViT-vs-CNN-MNIST/assets/40676497/b2e2d5dc-ee9c-4066-969b-3f6b5acfe904">


  
   

## Vision Transformer
Builds Vision Transformer by linearly projecting patches into encoder. We leverage a Pre-LN Transformer for better gradient flow and efficency. Monitored using TensorBoard.
- Ended with a lower test Accuracy of 93.81%.
- Training / Validation accuracy vs. epoch:
<img width="808" alt="Screenshot 2024-07-11 at 4 04 22 PM" src="https://github.com/shuh2002/ViT-vs-CNN-MNIST/assets/40676497/99c5d3bb-376b-4144-9bc7-de5286339a8d">
