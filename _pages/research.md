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

<p class="small">
<span style="font-weight:500"> <a name="exactline-wngt19-a" href="https://arxiv.org/abs/1810.11878" style="font-size: 12pt; color: #004D4D; text-decoration: none"> Unsupervised Evaluation Metrics and Learning Criteria for Non-Parallel Textual Transfer </a> <br> </span>
<span style="line-height:210%"> Richard Yuanzhe Pang, Kevin Gimpel </span>

<p style="font-size:10pt; margin-left: 30px"> <b> Abstract:</b> We consider the problem of automatically generating textual paraphrases with modified attributes or properties, focusing on the setting without parallel data (Hu et al., 2017; Shen et al., 2017). This setting poses challenges for evaluation. We show that the metric of post-transfer classification accuracy is insufficient on its own, and propose additional metrics based on semantic preservation and fluency as well as a way to combine them into a single overall score. We contribute new loss functions and training strategies to address the different metrics. Semantic preservation is addressed by adding a cyclic consistency loss and a loss based on paraphrase pairs, while fluency is improved by integrating losses based on style-specific language models. We experiment with a Yelp sentiment dataset and a new literature dataset that we propose, using multiple models that extend prior work (Shen et al., 2017). We demonstrate that our metrics correlate well with human judgments, at both the sentence-level and system-level. Automatic and manual evaluation also show large improvements over the baseline method of Shen et al. (2017). We hope that our proposed metrics can speed up system development for new textual transfer tasks while also encouraging the community to address our three complementary aspects of transfer quality.

<br><br>

[<a href="https://arxiv.org/pdf/1810.11878.pdf" style="color: #004D4D; text-decoration: none">paper</a>] [<a href="https://arxiv.org/pdf/1810.11878.pdf#page=11" style="color: #004D4D; text-decoration: none">supplementals</a>] [<a href="https://github.com/yzpang/textual-transfer-eval" style="color: #004D4D; text-decoration: none">code</a>] [<a href="../misc-files/pang+gimpel-textual-transfer-poster.pdf" style="color: #004D4D; text-decoration: none">poster</a>] [<a href="../misc-files/pang2018unsupervised.txt" style="color: #004D4D; text-decoration: none">old-BibTeX</a>]
</p>









<br>

<p class="small">
<span style="font-weight:500"> <a name="exactline-wngt19-b" href="https://arxiv.org/abs/1910.03747" style="font-size: 12pt; color: #004D4D; text-decoration: none"> The Daunting Task of Real-World Textual Style Transfer Auto-Evaluation </a> <br> </span>
<span style="line-height:210%"> Richard Yuanzhe Pang </span>

<p style="font-size:10pt; margin-left: 30px"> <b> Abstract:</b> The difficulty of textual style transfer lies in the lack of parallel corpora. Numerous advances have been proposed for the unsupervised generation. However, significant problems remain with the auto-evaluation of style transfer tasks. Based on the summary of Pang and Gimpel (2018) and Mir et al. (2019), style transfer evaluations rely on three criteria: style accuracy of transferred sentences, content similarity between original and transferred sentences, and fluency of transferred sentences. We elucidate the problematic current state of style transfer research. Given that current tasks do not represent real use cases of style transfer, current auto-evaluation approach is flawed. This discussion aims to bring researchers to think about the future of style transfer and style transfer evaluation research.

<br><br>

[<a href="https://arxiv.org/pdf/1910.03747.pdf" style="color: #004D4D; text-decoration: none">paper</a>] [<a href="../misc-files/pang-textual-transfer-problem-poster.pdf" style="color: #004D4D; text-decoration: none">poster</a>]
</p>





<br><br>



<p>









