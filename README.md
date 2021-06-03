# Attention is What You Get

*This is my entry into the [Bristol-Myers Squibb Molecular Translation](https://www.kaggle.com/c/bms-molecular-translation)  Kaggle competition. The notebook is publicly available at https://www.kaggle.com/c/bms-molecular-translation/discussion/url.*

Kagglers have coalesced around "Attention is What You Need" models, so I ask, *Is attention really all you need?*  This notebook include features to test that out: Enable/Disable CNN text feature extraction before the decoder self-attention; Increase model parameters without harming inference speed using decoder heads in series; and Experiment with my trainable & parallelizable alternative to beam search.

----
## Our Goal: Predict the "InChI" value of any given chemical compound diagram. 

International Chemical Identifiers ("InChI values") are a standardized encoding to describe chemical compounds. They take the form of a string of letters, numbers and deliminators, often between 100 - 400 characters long. 

The chemical diagrams are provided as PNG files, often of such low quality that it may take a human several seconds to decipher. 

Label length and image quality become a serious challenge here, because we must predict labels for a very large quantity of images. There are 1.6 million images in the test set abd 2.4 million images available in the training set!

----

## MODEL STRUCTURE: 

**Image CNN + Attention Image Encoder --> Attention + Dense Text Decoder.**

This is a hybrid approach with:
 
 - Image Encoder from [*Show, Attend and Tell: Neural Image Caption Generation with Visual Attention*](https://proceedings.mlr.press/v37/xuc15.pdf).  Generate image feature vectors using intermediate layer outputs from a pretrained CNN. (Here I use the more modern EfficientNet model with fixed weights and add a trainable Dense layer for customization.)
 
 - T2T encoder-decoder model from [*All You Need is Attention*](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) (Self-attention feature extraction for both encoder and decoder, joint encoder-decoder attention feature interactions, and an (optional) dense prediction output block. 

 - ***PLUS*** *(optional):* Decoder Output Blocks placed in Series (not stacked). Increase the number of trainable paramaters without adding inference computational complexity, while also allowing decoders to specialize on different regions of the output. (Note: Training is a bit trickier. My experiments show it is best to first train with the decoders using shared weights, then allowing them to vary later on in training.)
 
 - ***PLUS*** *(optional):* Is attention really all you need? Add a convolutional layer to enhance text features before decoder self-attention to experiment with performance differences with and without extra convolutional layer(s). Use of CNN's in NLP comes from [*Convolutional Sequence to Sequence Learning*](http://proceedings.mlr.press/v70/gehring17a.html.)

 - ***PLUS*** *(optional):* Beam-Search Alternative, an extra decoding layer applied after the full logits prediction has been made. This takes the form of a bidirectional RNN applied to the full logits sequence. Because a full (initial) prediction has already been made, computations can be paralelized using stateful RNNs. (See more details below.)

*Optional features can be enabled/disabled using parameters in my model definitions.*

----

## NEXT STEPS:

 - Implement **TPU training**. (Currently runs on GPU. Note that a CPU along is not enough to achieve acceptable inference speed.)

 - experiment with **"Tokens-to-Token ViT"** in place of the image CNN. (Technique from [*Training Vision Transformers from Scratch on ImageNet*](https://arxiv.org/pdf/2101.11986.pdf)
  
 - Train my **Beam-search Alternative**. 

    - Beam search is a technique to modify model predictions to reflect the (local) maximum likelihood estimate. However, it is *very* local in that computation expense increases quickly with the number of character steps taken into account. This is also a hard-coded algorithm, which is somewhat contrary to the philosophy of deep learning.

    - A *Beam-search Alternative* would be an extra decoding layer applied *after* the full logits prediction has been made. This might be in the form of a stateful, bidirectional RNN that is computationally parallizable because it is applied to the full logits sequence.

    - This is coded and ready to train, although I have not yet had the time to do so.

 - Treat the number of convolutional layers (decoder feature extraction) and number of decoders places in series (decoder prediction output) as **new hyperparamaters** to tune.

----

## CITATIONS

- "Attention is All You Need." 
 - Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin. NIPS (2017). *https://research.google/pubs/pub46201/*

- "Convolutional Sequence to Sequence Learning."
 
  - Gehring, J., Auli, M., Grangier, D., Yarats, D. & Dauphin, Y.N.. (2017). Convolutional Sequence to Sequence Learning. Proceedings of the 34th International Conference on Machine Learning, in Proceedings of Machine Learning Research 70:1243-1252 Available from *http://proceedings.mlr.press/v70/gehring17a.html.*


-  "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention."
  -  Xu, K., Ba, J., Kiros, R., Cho, K., Courville, A., Salakhudinov, R., Zemel, R. & Bengio, Y.. (2015). Show, Attend and Tell: Neural Image Caption Generation with Visual Attention. Proceedings of the 32nd International Conference on Machine Learning, in Proceedings of Machine Learning Research 37:2048-2057 Available from *http://proceedings.mlr.press/v37/xuc15.html.* 
            

- "Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet"

  - Li Yuan, Yunpeng Chen, Tao Wang, Weihao Yu, Yujun Shi, Zihang Jiang, Francis EH Tay, Jiashi Feng, Shuicheng Yan. Preprint (2021). Available at *https://arxiv.org/abs/2101.11986*.

- Special thanks to [Darien Schettler](https://www.kaggle.com/dschettler8845/bms-efficientnetv2-tpu-e2e-pipeline-in-3hrs/notebook.) for leading readers to the "Show" and "Attention" papers cited above, and sharing his progress at various stages in the competition through public Kaggle notebooks. My work includes a few of his hyperparameter choices and selection of EfficientNet transfer model. I did not read or use his implementations of the papers above.

- It is possible my idea of a Beam Search Alternative is based on a lecture video from DeepLearning.ai's [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)  on Coursera.

- **Dataset / Kaggle Competition:** "Bristol-Myers Squibb – Molecular Translation" competition on Kaggle (2021). *https://www.kaggle.com/c/bms-molecular-translation*
