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
<span style="font-weight:500"> <a name="exactline-consistency-rlm" href="https://arxiv.org/abs/2002.02492" style="font-size: 12pt; color: #005F5F; text-decoration: none"> Consistency of a Recurrent Language Model With Respect to Incomplete Decoding </a> <br> </span>
<span style="line-height:170%"> Sean Welleck, Ilia Kulikov, Jaedeok Kim, Richard Yuanzhe Pang, Kyunghyun Cho </span>

<p style="font-size:10pt; margin-left: 30px"> <b> Abstract:</b> Despite strong performance on a variety of tasks, neural sequence models trained with maximum likelihood have been shown to exhibit issues such as length bias and degenerate repetition. We study the related issue of receiving infinite-length sequences from a recurrent language model when using common decoding algorithms. To analyze this issue, we first define inconsistency of a decoding algorithm, meaning that the algorithm can yield an infinite-length sequence that has zero probability under the model. We prove that commonly used incomplete decoding algorithms -- greedy search, beam search, top-k sampling, and nucleus sampling -- are inconsistent, despite the fact that recurrent language models are trained to produce sequences of finite length. Based on these insights, we propose two remedies which address inconsistency: consistent variants of top-k and nucleus sampling, and a self-terminating recurrent language model. Empirical results show that inconsistency occurs in practice, and that the proposed methods prevent inconsistency.

<br><br>

[<a href="https://arxiv.org/pdf/2002.02492.pdf" style="color: #005F5F; text-decoration: none">paper</a>] [<a href="../misc-files/welleck2020consistency.txt" style="color: #005F5F; text-decoration: none">BibTeX</a>]
</p>





<br>

<p class="small">
<span style="font-weight:500"> <a name="exactline-spen" href="https://arxiv.org/abs/1911.02891" style="font-size: 12pt; color: #004A90; text-decoration: none"> Improving Joint Training of Inference Networks and Structured Prediction Energy Networks </a> <br> </span>
<span style="line-height:210%"> Lifu Tu, Richard Yuanzhe Pang, Kevin Gimpel </span>

<p style="font-size:10pt; margin-left: 30px"> <b> Abstract:</b> Deep energy-based models are powerful, but pose challenges for learning and inference (Belanger and McCallum, 2016). Tu and Gimpel (2018) developed an efficient framework for energy-based models by training "inference networks" to approximate structured inference instead of using gradient descent. However, their alternating optimization approach suffers from instabilities during training, requiring additional loss terms and careful hyperparameter tuning. In this paper, we contribute several strategies to stabilize and improve this joint training of energy functions and inference networks for structured prediction. We design a compound objective to jointly train both cost-augmented and test-time inference networks along with the energy function. We propose joint parameterizations for the inference networks that encourage them to capture complementary functionality during learning. We empirically validate our strategies on two sequence labeling tasks, showing easier paths to strong performance than prior work, as well as further improvements with global energy terms.

<br><br>

[<a href="https://arxiv.org/pdf/1911.02891.pdf" style="color: #004A90; text-decoration: none">paper</a>] [<a href="../misc-files/tu2019improving.txt" style="color: #004A90; text-decoration: none">BibTex</a>] 
</p>





<br>


<p class="small">
<span style="font-weight:500"> <a style="font-size: 12pt; color: #6F4195; text-decoration: none"> ENGINE: Energy-Based Inference Networks for Non-Autoregressive Machine Translation </a> <br> </span>
<span style="line-height:210%"> Lifu Tu, Richard Yuanzhe Pang, Sam Wiseman, Kevin Gimpel</span>
<br>
<span style="line-height:170%; font-size: 10.5pt"> In <i>Proceedings of ACL 2020</i> </span>

<p style="font-size:10pt; margin-left: 30px">
[<a style="color: #6F4195; text-decoration: none">paper (soon)</a>] 
</p>






<br>


<p class="small">
<span style="font-weight:500"> <a style="font-size: 12pt; color: #22789D; text-decoration: none"> Intermediate-Task Transfer Learning with Pretrained Language Models: When and Why Does It Work? </a> <br> </span>
<span style="line-height:160%"> Yada Pruksachatkun, Jason Phang, Haokun Liu, Phu Mon Htut, Xiaoyi Zhang, Richard Yuanzhe Pang, Clara Vania, Katharina Kann, Samuel R. Bowman</span>
<br>
<span style="line-height:180%; font-size: 10.5pt"> In <i>Proceedings of ACL 2020</i> </span>

<p style="font-size:10pt; margin-left: 30px"> 
[<a style="color: #22789D; text-decoration: none">paper (soon)</a>] 
</p>












<br>

<p class="small">
<span style="font-weight:500"> <a name="exactline-wngt19-a" href="https://arxiv.org/abs/1810.11878" style="font-size: 12pt; color: #005F5F; text-decoration: none"> Unsupervised Evaluation Metrics and Learning Criteria for Non-Parallel Textual Transfer </a> <br> </span>
<span style="line-height:210%"> Richard Yuanzhe Pang, Kevin Gimpel </span>
<br>
<span style="line-height:170%; font-size: 10.5pt"> In <i>Proceedings of EMNLP 2019 Workshop on Neural Generation and Translation (WNGT)</i></span>

<p style="font-size:10pt; margin-left: 30px"> <b> Abstract:</b> We consider the problem of automatically generating textual paraphrases with modified attributes or properties, focusing on the setting without parallel data (Hu et al., 2017; Shen et al., 2017). This setting poses challenges for evaluation. We show that the metric of post-transfer classification accuracy is insufficient on its own, and propose additional metrics based on semantic preservation and fluency as well as a way to combine them into a single overall score. We contribute new loss functions and training strategies to address the different metrics. Semantic preservation is addressed by adding a cyclic consistency loss and a loss based on paraphrase pairs, while fluency is improved by integrating losses based on style-specific language models. We experiment with a Yelp sentiment dataset and a new literature dataset that we propose, using multiple models that extend prior work (Shen et al., 2017). We demonstrate that our metrics correlate well with human judgments, at both the sentence-level and system-level. Automatic and manual evaluation also show large improvements over the baseline method of Shen et al. (2017). We hope that our proposed metrics can speed up system development for new textual transfer tasks while also encouraging the community to address our three complementary aspects of transfer quality.

<br><br>

[<a href="https://arxiv.org/pdf/1810.11878.pdf" style="color: #005F5F; text-decoration: none">paper</a>] [<a href="https://arxiv.org/pdf/1810.11878.pdf#page=11" style="color: #005F5F; text-decoration: none">supplementals</a>] [<a href="https://github.com/yzpang/textual-transfer-eval" style="color: #005F5F; text-decoration: none">code</a>] [<a href="../misc-files/pang+gimpel-textual-transfer-poster.pdf" style="color: #005F5F; text-decoration: none">poster</a>] [<a href="../misc-files/pang2018unsupervised.txt" style="color: #005F5F; text-decoration: none">BibTeX</a>]
</p>









<br>

<p class="small">
<span style="font-weight:500"> <a name="exactline-wngt19-b" href="https://arxiv.org/abs/1910.03747" style="font-size: 12pt; color: #005F5F; text-decoration: none"> The Daunting Task of Real-World Textual Style Transfer Auto-Evaluation </a> <br> </span>
<span style="line-height:210%"> Richard Yuanzhe Pang </span>
<br>
<span style="line-height:170%; font-size: 10.5pt"> Extended abstract in <i>EMNLP 2019 Workshop on Neural Generation and Translation (WNGT)</i>; abstract in <i>Proceedings of the Workshop on Noisy User-generated Text (W-NUT)</i></span> 

<p style="font-size:10pt; margin-left: 30px"> <b> Abstract:</b> The difficulty of textual style transfer lies in the lack of parallel corpora. Numerous advances have been proposed for the unsupervised generation. However, significant problems remain with the auto-evaluation of style transfer tasks. Based on the summary of Pang and Gimpel (2018) and Mir et al. (2019), style transfer evaluations rely on three criteria: style accuracy of transferred sentences, content similarity between original and transferred sentences, and fluency of transferred sentences. We elucidate the problematic current state of style transfer research. Given that current tasks do not represent real use cases of style transfer, current auto-evaluation approach is flawed. This discussion aims to bring researchers to think about the future of style transfer and style transfer evaluation research.

<br><br>

[<a href="https://arxiv.org/pdf/1910.03747.pdf" style="color: #005F5F; text-decoration: none">paper</a>] [<a href="../misc-files/pang-textual-transfer-problem-poster.pdf" style="color: #005F5F; text-decoration: none">poster</a>] [<a href="../misc-files/pang2019daunting.txt" style="color: #005F5F; text-decoration: none">BibTeX</a>]
</p>





<br><br>



<p>









