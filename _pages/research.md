---
title:  "Research"
layout: archive
permalink: /research/
author_profile: true
comments: true
---



<html>
<head>



<style>
p.xsmall {
    line-height: 1.55;
    font-size: 9.5pt;
    margin-left: 0px; 
}

p.small {
    line-height: 1.55;
    font-size: 11.5pt;
    margin-left: 0px; 
}

p.small2 {
    line-height: 2.00;
    font-size: 11.5pt;
    margin-left: 0px; 
}

p.medium {
    line-height: 1.55;
    font-size: 12.5pt;
    margin-left: 0px; 
}

p.big {
    line-height: 1.55;
}

p.noindent {
    line-height: 1.55;
    font-size: 13pt;
}


</style>
</head>


<body>






<br>

<!-- <font size="3"> -->
<p class="small">
<span style="font-weight:500"> <a name="exactline-off-policy-gen" href="https://arxiv.org/abs/2009.07839" style="font-size: 12pt; color: #023DB4; text-decoration: none"> Text Generation by Learning from Demonstrations</a> <br> </span>
<span style="line-height:160%"> Richard Yuanzhe Pang, He He </span>
<br>
<span style="line-height:180%; font-size: 10.5pt"> In <i>Proceedings of ICLR 2021</i> </span>

<p style="font-size:10pt; margin-left: 30px"> <b> Abstract:</b> Current approaches to text generation largely rely on autoregressive models and maximum likelihood estimation. This paradigm leads to (i) diverse but low-quality samples due to mismatched learning objective and evaluation metric (likelihood vs. quality) and (ii) exposure bias due to mismatched history distributions (gold vs. model-generated). To alleviate these problems, we frame text generation as an offline reinforcement learning (RL) problem with expert demonstrations (i.e., the reference), where the goal is to maximize quality given model-generated histories. We propose GOLD (generation by off-policy learning from demonstrations): an easy-to-optimize algorithm that learns from the demonstrations by importance weighting. Intuitively, GOLD upweights confident tokens and downweights unconfident ones in the reference during training, avoiding optimization issues faced by prior RL approaches that rely on online data collection. According to both automatic and human evaluation, models trained by GOLD outperform those trained by MLE and policy gradient on summarization, question generation, and machine translation. Further, our models are less sensitive to decoding algorithms and alleviate exposure bias.

<br><br>

[<a href="https://arxiv.org/pdf/2009.07839.pdf" style="color: #023DB4; text-decoration: none">paper</a>] [<a href="https://openreview.net/forum?id=RovX-uQ1Hua" style="color: #023DB4; text-decoration: none">openreview</a>] [<a href="https://github.com/yzpang/gold-off-policy-text-gen-iclr21" style="color: #023DB4; text-decoration: none">code</a>] [<a href="../misc-files/bibs/pang2021text.txt" style="color: #023DB4; text-decoration: none">bibtex</a>]



</p>



<br>

<p class="small">
<span style="font-weight:500"> <a name="exactline-spen" href="https://arxiv.org/abs/1911.02891" style="font-size: 12pt; color: #0A897D; text-decoration: none"> Improving Joint Training of Inference Networks and Structured Prediction Energy Networks </a> <br> </span>
<span style="line-height:160%"> Lifu Tu, Richard Yuanzhe Pang, Kevin Gimpel </span>
<br>
<span style="line-height:180%; font-size: 10.5pt"> In <i>Proceedings of EMNLP 2020 Workshop on Structured Prediction for NLP (SPNLP)</i>; spotlight paper </span>

<p style="font-size:10pt; margin-left: 30px"> <b> Abstract:</b> Deep energy-based models are powerful, but pose challenges for learning and inference (Belanger and McCallum, 2016). Tu and Gimpel (2018) developed an efficient framework for energy-based models by training "inference networks" to approximate structured inference instead of using gradient descent. However, their alternating optimization approach suffers from instabilities during training, requiring additional loss terms and careful hyperparameter tuning. In this paper, we contribute several strategies to stabilize and improve this joint training of energy functions and inference networks for structured prediction. We design a compound objective to jointly train both cost-augmented and test-time inference networks along with the energy function. We propose joint parameterizations for the inference networks that encourage them to capture complementary functionality during learning. We empirically validate our strategies on two sequence labeling tasks, showing easier paths to strong performance than prior work, as well as further improvements with global energy terms.

<br><br>

[<a href="https://arxiv.org/pdf/1911.02891.pdf" style="color: #0A897D; text-decoration: none">paper</a>] [<a href="../misc-files/bibs/tu2019improving.txt" style="color: #0A897D; text-decoration: none">bibtex</a>] 
</p>





<br>

<!-- <font size="3"> -->
<p class="small">
<span style="font-weight:500"> <a name="exactline-emnlp20-a" href="https://arxiv.org/abs/2002.02492" style="font-size: 12pt; color: #0A897D; text-decoration: none"> Consistency of a Recurrent Language Model With Respect to Incomplete Decoding </a> <br> </span>
<span style="line-height:160%"> Sean Welleck, Ilia Kulikov, Jaedeok Kim, Richard Yuanzhe Pang, Kyunghyun Cho </span>
<br>
<span style="line-height:180%; font-size: 10.5pt"> In <i>Proceedings of EMNLP 2020</i>; also appearing in the non-archival <i>DeepMath 2020</i> </span>

<p style="font-size:10pt; margin-left: 30px"> <b> Abstract:</b> Despite strong performance on a variety of tasks, neural sequence models trained with maximum likelihood have been shown to exhibit issues such as length bias and degenerate repetition. We study the related issue of receiving infinite-length sequences from a recurrent language model when using common decoding algorithms. To analyze this issue, we first define inconsistency of a decoding algorithm, meaning that the algorithm can yield an infinite-length sequence that has zero probability under the model. We prove that commonly used incomplete decoding algorithms -- greedy search, beam search, top-k sampling, and nucleus sampling -- are inconsistent, despite the fact that recurrent language models are trained to produce sequences of finite length. Based on these insights, we propose two remedies which address inconsistency: consistent variants of top-k and nucleus sampling, and a self-terminating recurrent language model. Empirical results show that inconsistency occurs in practice, and that the proposed methods prevent inconsistency.

<br><br>

[<a href="https://arxiv.org/pdf/2002.02492.pdf" style="color: #0A897D; text-decoration: none">paper</a>] [<a href="https://github.com/uralik/consistency-lm" style="color: #0A897D; text-decoration: none">code</a>] [<a href="../misc-files/bibs/welleck2020consistency.txt" style="color: #0A897D; text-decoration: none">bibtex</a>]
</p>





<br>


<p class="small">
<span style="font-weight:500"> <a href="https://arxiv.org/abs/2005.00850" name="exactline-acl20-a" style="font-size: 12pt; color: #753DA4; text-decoration: none"> ENGINE: Energy-Based Inference Networks for Non-Autoregressive Machine Translation </a> <br> </span>
<span style="line-height:160%"> Lifu Tu, Richard Yuanzhe Pang, Sam Wiseman, Kevin Gimpel</span>
<br>
<span style="line-height:180%; font-size: 10.5pt"> In <i>Proceedings of ACL 2020</i> </span>

<p style="font-size:10pt; margin-left: 30px"> <b> Abstract:</b> We propose to train a non-autoregressive machine translation model to minimize the energy defined by a pretrained autoregressive model. In particular, we view our non-autoregressive translation system as an inference network (Tu and Gimpel, 2018) trained to minimize the autoregressive teacher energy. This contrasts with the popular approach of training a non-autoregressive model on a distilled corpus consisting of the beam-searched outputs of such a teacher model. Our approach, which we call ENGINE (ENerGy-based Inference NEtworks), achieves state-of-the-art non-autoregressive results on the IWSLT 2014 DE-EN and WMT 2016 RO-EN datasets, approaching the performance of autoregressive models.

<br><br>

[<a href="https://arxiv.org/pdf/2005.00850.pdf" style="color: #753DA4; text-decoration: none">paper</a>] [<a href="https://github.com/lifu-tu/ENGINE" style="color: #753DA4; text-decoration: none">code</a>] [<a href="../misc-files/bibs/tu2020engine.txt" style="color: #753DA4; text-decoration: none">bibtex</a>]
</p>






<br>


<p class="small">
<span style="font-weight:500"> <a href="https://arxiv.org/abs/2005.00628" name="exactline-acl20-b" style="font-size: 12pt; color: #22789D; text-decoration: none"> Intermediate-Task Transfer Learning with Pretrained Language Models: When and Why Does It Work? </a> <br> </span>
<span style="line-height:160%"> Yada Pruksachatkun, Jason Phang, Haokun Liu, Phu Mon Htut, Xiaoyi Zhang, Richard Yuanzhe Pang, Clara Vania, Katharina Kann, Samuel R. Bowman</span>
<br>
<span style="line-height:180%; font-size: 10.5pt"> In <i>Proceedings of ACL 2020</i> </span>

<p style="font-size:10pt; margin-left: 30px"> <b> Abstract:</b> While pretrained models such as BERT have shown large gains across natural language understanding tasks, their performance can be improved by further training the model on a data-rich intermediate task, before fine-tuning it on a target task. However, it is still poorly understood when and why intermediate-task training is beneficial for a given target task. To investigate this, we perform a large-scale study on the pretrained RoBERTa model with 110 intermediate-target task combinations. We further evaluate all trained models with 25 probing tasks meant to reveal the specific skills that drive transfer. We observe that intermediate tasks requiring high-level inference and reasoning abilities tend to work best. We also observe that target task performance is strongly correlated with higher-level abilities such as coreference resolution. However, we fail to observe more granular correlations between probing and target task performance, highlighting the need for further work on broad-coverage probing benchmarks. We also observe evidence that the forgetting of knowledge learned during pretraining may limit our analysis, highlighting the need for further work on transfer learning methods in these settings.

<br><br>

[<a href="https://arxiv.org/pdf/2005.00628.pdf" style="color: #22789D; text-decoration: none">paper</a>] [<a href="../misc-files/bibs/pruksachatkun2020intermediate.txt" style="color: #22789D; text-decoration: none">bibtex</a>]
</p>








<br>

<p class="small">
<span style="font-weight:500"> <a name="exactline-wngt19-a" href="https://arxiv.org/abs/1810.11878" style="font-size: 12pt; color: #023DB4; text-decoration: none"> Unsupervised Evaluation Metrics and Learning Criteria for Non-Parallel Textual Transfer </a> <br> </span>
<span style="line-height:160%"> Richard Yuanzhe Pang, Kevin Gimpel </span>
<br>
<span style="line-height:180%; font-size: 10.5pt"> In <i>Proceedings of EMNLP 2019 Workshop on Neural Generation and Translation (WNGT)</i></span>

<p style="font-size:10pt; margin-left: 30px"> <b> Abstract:</b> We consider the problem of automatically generating textual paraphrases with modified attributes or properties, focusing on the setting without parallel data (Hu et al., 2017; Shen et al., 2017). This setting poses challenges for evaluation. We show that the metric of post-transfer classification accuracy is insufficient on its own, and propose additional metrics based on semantic preservation and fluency as well as a way to combine them into a single overall score. We contribute new loss functions and training strategies to address the different metrics. Semantic preservation is addressed by adding a cyclic consistency loss and a loss based on paraphrase pairs, while fluency is improved by integrating losses based on style-specific language models. We experiment with a Yelp sentiment dataset and a new literature dataset that we propose, using multiple models that extend prior work (Shen et al., 2017). We demonstrate that our metrics correlate well with human judgments, at both the sentence-level and system-level. Automatic and manual evaluation also show large improvements over the baseline method of Shen et al. (2017). We hope that our proposed metrics can speed up system development for new textual transfer tasks while also encouraging the community to address our three complementary aspects of transfer quality.

<br><br>

[<a href="https://arxiv.org/pdf/1810.11878.pdf" style="color: #023DB4; text-decoration: none">paper</a>] [<a href="https://arxiv.org/pdf/1810.11878.pdf#page=11" style="color: #023DB4; text-decoration: none">supplementals</a>] [<a href="../misc-files/pang+gimpel-textual-transfer-poster.pdf" style="color: #023DB4; text-decoration: none">poster</a>] [<a href="../misc-files/bibs/pang2018unsupervised.txt" style="color: #023DB4; text-decoration: none">bibtex</a>]
</p>









<br>

<p class="small">
<span style="font-weight:500"> <a name="exactline-wngt19-b" href="https://arxiv.org/abs/1910.03747" style="font-size: 12pt; color: #023DB4; text-decoration: none"> The Daunting Task of Real-World Textual Style Transfer Auto-Evaluation </a> <br> </span>
<span style="line-height:180%"> Richard Yuanzhe Pang </span>
<br>
<span style="line-height:140%; font-size: 10.5pt"> Extended abstract in <i>EMNLP 2019 Workshop on Neural Generation and Translation (WNGT)</i>; abstract in <i>Proceedings of the Workshop on Noisy User-generated Text (W-NUT)</i></span> 

<p style="font-size:10pt; margin-left: 30px"> <b> Abstract:</b> The difficulty of textual style transfer lies in the lack of parallel corpora. Numerous advances have been proposed for the unsupervised generation. However, significant problems remain with the auto-evaluation of style transfer tasks. Based on the summary of Pang and Gimpel (2018) and Mir et al. (2019), style transfer evaluations rely on three criteria: style accuracy of transferred sentences, content similarity between original and transferred sentences, and fluency of transferred sentences. We elucidate the problematic current state of style transfer research. Given that current tasks do not represent real use cases of style transfer, current auto-evaluation approach is flawed. This discussion aims to bring researchers to think about the future of style transfer and style transfer evaluation research.

<br><br>

[<a href="https://arxiv.org/pdf/1910.03747.pdf" style="color: #023DB4; text-decoration: none">paper</a>] [<a href="../misc-files/pang-textual-transfer-problem-poster.pdf" style="color: #023DB4; text-decoration: none">poster</a>] [<a href="../misc-files/bibs/pang2019daunting.txt" style="color: #023DB4; text-decoration: none">bibtex</a>]
</p>





<br><br>



<p>









