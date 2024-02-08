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
<!-- <span style="line-height:170%"> 
<i>Primary</i> research subfields: <a style="color: #245ED2; text-decoration: none">text generation</a> (including <a style="color: #753DA4; text-decoration: none">machine translation</a>) and <a style="color: #245ED2; text-decoration: none">learning from rewards</a>, <a style="color: #22789D; text-decoration: none">language understanding</a>, <a style="color: #439371; text-decoration: none">reasoning</a>, and <a style="color: #757575; text-decoration: none">others</a>. 

<br>  -->

[<a href="https://scholar.google.com/citations?hl=en&user=vg_IkckAAAAJ" style="color: #4C8BF5; text-decoration: none" target="_blank">google scholar</a>] [<a href="https://www.semanticscholar.org/author/Richard-Yuanzhe-Pang/46230016" style="color: #4C8BF5; text-decoration: none" target="_blank">semantic scholar</a>] [<a href="https://dblp.org/pid/250/9059.html" style="color: #4C8BF5; text-decoration: none" target="_blank">dblp</a>] [<a href="../misc-files/abbreviations.txt" style="color: #4C8BF5; text-decoration: none" target="_blank">abbreviations</a>]

<!-- </span> -->


<br><br>



<p class="small">
<span style="line-height:170%"> <u> Overview of selected research directions </u> </span>



<ul class="small" style="font-size: 11.5pt">
<li> <b>Learning from rewards in text generation</b>: <a href="https://arxiv.org/abs/2009.07839" style="color: #245ED2; text-decoration: none">GOLD</a> (offline RL), <a href="https://arxiv.org/abs/2112.08670" style="color: #245ED2; text-decoration: none">amortized noisy channel NMT</a> (off-policy RL & knowledge distillation), <a href="https://arxiv.org/abs/2211.08714" style="color: #245ED2; text-decoration: none">reward gaming</a> (on-policy RL), <a href="https://arxiv.org/abs/2106.02278" style="color: #245ED2; text-decoration: none">AgreeSum</a> (on-policy RL for multi-doc summarization), <a href="https://arxiv.org/abs/2005.00850" style="color: #245ED2; text-decoration: none">ENGINE for non-autoregressive NMT</a> ("soft" knowledge distillation), <a href="https://arxiv.org/abs/2307.14117" style="color: #245ED2; text-decoration: none">implicit feedback in dialogue</a> (extracting implicit reward from deployment data), <a href="https://arxiv.org/abs/2401.10020" style="color: #245ED2; text-decoration: none">self-rewarding LLM</a>, etc.</li>

<li> <b>Reasoning</b>: <a href="https://arxiv.org/abs/2305.15269" style="color: #439371; text-decoration: none">PrOntoQA-OOD</a> (deductive reasoning); I'm currently working on improving reasoning capabilities in general </li>

<li> <b>Scalable oversight benchmarks</b>: <a href="https://arxiv.org/abs/2112.08608" style="color: #22789D; text-decoration: none">QuALITY</a> (long-document QA; related: <a href="https://arxiv.org/abs/2205.11465" style="color: #22789D; text-decoration: none">SQuALITY</a> long-document summarization), <a href="https://arxiv.org/abs/2311.12022" style="color: #22789D; text-decoration: none"> GPQA</a> (graduate-level Google-proof QA); I'm working on developing methods as well </li>

</ul>
</p>


<p class="small">
<span style="line-height:170%"> <u> Publications and preprints (2023-) </u> </span>


<p class="small">
<span style="font-weight:500"> <a name="exactline-self-rewarding-lm" href="https://arxiv.org/abs/2401.10020" style="font-size: 12pt; color: #245ED2; text-decoration: none"> Self-Rewarding Language Models </a> <br> </span>
<span style="line-height:160%"> Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Sainbayar Sukhbaatar, Jing Xu, Jason Weston </span>
<br>
<span style="line-height:180%; font-size: 10.5pt"> <i>Preprint</i>, January 2024 </span>

<p style="font-size:10pt; margin-left: 15px; line-height:140%"> <b> Abstract:</b> We posit that to achieve superhuman agents, future models require superhuman feedback in order to provide an adequate training signal. Current approaches commonly train reward models from human preferences, which may then be bottlenecked by human performance level, and secondly these separate frozen reward models cannot then learn to improve during LLM training. In this work, we study Self-Rewarding Language Models, where the language model itself is used via LLM-as-a-Judge prompting to provide its own rewards during training. We show that during Iterative DPO training that not only does instruction following ability improve, but also the ability to provide high-quality rewards to itself. Fine-tuning Llama 2 70B on three iterations of our approach yields a model that outperforms many existing systems on the AlpacaEval 2.0 leaderboard, including Claude 2, Gemini Pro, and GPT-4 0613. While only a preliminary study, this work opens the door to the possibility of models that can continually improve in both axes.

<br><br>

[<a href="https://arxiv.org/pdf/2401.10020.pdf" style="color: #245ED2; text-decoration: none">paper</a>] [<a href="../misc-files/bibs/yuan2024self.txt" style="color: #245ED2; text-decoration: none">bibtex</a>] | by others: [<a href="../misc-files/links/self-reward-articles.html" style="color: #245ED2; text-decoration: none">press, articles, videos</a>]


<br><br>


<p class="small">
<span style="font-weight:500"> <a name="exactline-dialogue-implicit-feedback" href="https://arxiv.org/abs/2307.14117" style="font-size: 12pt; color: #245ED2; text-decoration: none"> Leveraging Implicit Feedback from Deployment Data in Dialogue </a> <br> </span>
<span style="line-height:160%"> Richard Yuanzhe Pang, Stephen Roller, Kyunghyun Cho, He He, Jason Weston </span>
<br>
<span style="line-height:180%; font-size: 10.5pt"> In <i>Proceedings of EACL 2024</i> </span>

<p style="font-size:10pt; margin-left: 15px; line-height:140%"> <b> Abstract:</b> We study improving social conversational agents by learning from natural dialogue between users and a deployed model, without extra annotations. To implicitly measure the quality of a machine-generated utterance, we leverage signals like user response length, sentiment and reaction of the future human utterances in the collected dialogue episodes. Our experiments use the publicly released deployment data from BlenderBot (Xu et al., 2023). Human evaluation indicates improvements in our new models over baseline responses; however, we find that some proxy signals can lead to more generations with undesirable properties as well. For example, optimizing for conversation length can lead to more controversial or unfriendly generations compared to the baseline, whereas optimizing for positive sentiment or reaction can decrease these behaviors. 

<br><br>

[<a href="https://arxiv.org/pdf/2307.14117.pdf" style="color: #245ED2; text-decoration: none">paper</a>] [<a href="../misc-files/bibs/pang2023leveraging.txt" style="color: #245ED2; text-decoration: none">bibtex</a>]


<br><br>


<p class="small">
<span style="font-weight:500"> <a name="exactline-gpqa" href="https://arxiv.org/abs/2311.12022" style="font-size: 12pt; color: #22789D; text-decoration: none"> GPQA: A Graduate-Level Google-Proof Q&A Benchmark </a> <br> </span>
<span style="line-height:160%"> David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, Samuel R. Bowman </span>
<br>
<span style="line-height:180%; font-size: 10.5pt"> <i>Preprint</i>, November 2023 </span>

<p style="font-size:10pt; margin-left: 15px; line-height:140%"> <b> Abstract:</b> We present GPQA, a challenging dataset of 448 multiple-choice questions written by domain experts in biology, physics, and chemistry. We ensure that the questions are high-quality and extremely difficult: experts who have or are pursuing PhDs in the corresponding domains reach 65% accuracy (74% when discounting clear mistakes the experts identified in retrospect), while highly skilled non-expert validators only reach 34% accuracy, despite spending on average over 30 minutes with unrestricted access to the web (i.e., the questions are "Google-proof"). The questions are also difficult for state-of-the-art AI systems, with our strongest GPT-4 based baseline achieving 39% accuracy. If we are to use future AI systems to help us answer very hard questions, for example, when developing new scientific knowledge, we need to develop scalable oversight methods that enable humans to supervise their outputs, which may be difficult even if the supervisors are themselves skilled and knowledgeable. The difficulty of GPQA both for skilled non-experts and frontier AI systems should enable realistic scalable oversight experiments, which we hope can help devise ways for human experts to reliably get truthful information from AI systems that surpass human capabilities. 
<br><br>

[<a href="https://arxiv.org/pdf/2311.12022.pdf" style="color: #22789D; text-decoration: none">paper</a>] [<a href="https://github.com/idavidrein/gpqa" style="color: #22789D; text-decoration: none">data & code</a>] [<a href="../misc-files/bibs/rein2023gpqa.txt" style="color: #22789D; text-decoration: none">bibtex</a>] | by others: [<a href="https://www.youtube.com/watch?v=e4jOGywUCc4" style="color: #22789D; text-decoration: none">video mention</a>] 


<br><br>


<p class="small">
<span style="font-weight:500"> <a name="exactline-reasoning" href="https://arxiv.org/abs/2305.15269" style="font-size: 12pt; color: #439371; text-decoration: none"> Testing the General Deductive Reasoning Capacity of Large Language Models Using OOD Examples </a> <br> </span>
<span style="line-height:160%"> Abulhair Saparov, Richard Yuanzhe Pang, Vishakh Padmakumar, Nitish Joshi, Seyed Mehran Kazemi, Najoung Kim*, He He* </span>
<br>
<span style="line-height:180%; font-size: 10.5pt"> In <i>Proceedings of NeurIPS 2023</i> </span>

<p style="font-size:10pt; margin-left: 15px; line-height:140%"> <b> Abstract:</b> Given the intractably large size of the space of proofs, any model that is capable of general deductive reasoning must generalize to proofs of greater complexity. Recent studies have shown that large language models (LLMs) possess some abstract deductive reasoning ability given chain-of-thought prompts. However, they have primarily been tested on proofs using modus ponens or of a specific size, and from the same distribution as the in-context examples. To measure the general deductive reasoning ability of LLMs, we test on a broad set of deduction rules and measure their ability to generalize to more complex proofs from simpler demonstrations from multiple angles: depth-, width-, and compositional generalization. To facilitate systematic exploration, we construct a new synthetic and programmable reasoning dataset that enables control over deduction rules and proof complexity. Our experiments on four LLMs of various sizes and training objectives show that they are able to generalize to longer and compositional proofs. However, they require explicit demonstrations to produce hypothetical subproofs, specifically in proof by cases and proof by contradiction. 

<br><br>

[<a href="https://arxiv.org/pdf/2305.15269.pdf" style="color: #439371; text-decoration: none">paper</a>] [<a href="../misc-files/pang-reasoning-poster-icml-klr-workshop-2023.pdf" style="color: #439371; text-decoration: none">poster at <i>ICML 2023 Knowledge and Logical Reasoning Workshop</i></a>] [<a href="../misc-files/bibs/saparov2023testing.txt" style="color: #439371; text-decoration: none">bibtex</a>]


<p class="small">
<span style="font-weight:500"> <a name="exactline-icml23" href="https://arxiv.org/abs/2303.04562" style="font-size: 12pt; color: #245ED2; text-decoration: none"> Extrapolative Controlled Sequence Generation via Iterative Refinement </a> <br> </span>
<span style="line-height:160%"> Vishakh Padmakumar, Richard Yuanzhe Pang, He He, Ankur P. Parikh </span>
<br>
<span style="line-height:180%; font-size: 10.5pt"> In <i>Proceedings of ICML 2023</i> </span>

<p style="font-size:10pt; margin-left: 15px; line-height:140%"> <b> Abstract:</b> We study the problem of extrapolative controlled generation, i.e., generating sequences with attribute values beyond the range seen in training. This task is of significant importance in automated design, especially drug discovery, where the goal is to design novel proteins that are better (e.g., more stable) than existing sequences. Thus, by definition, the target sequences and their attribute values are out of the training distribution, posing challenges to existing methods that aim to directly generate the target sequence. Instead, in this work, we propose Iterative Controlled Extrapolation (ICE) which iteratively makes local edits to a sequence to enable extrapolation. We train the model on synthetically generated sequence pairs that demonstrate small improvement in the attribute value. Results on one natural language task (sentiment analysis) and two protein engineering tasks (ACE2 stability and AAV fitness) show that ICE considerably outperforms state-of-the-art approaches despite its simplicity. 

<br><br>

[<a href="https://arxiv.org/pdf/2303.04562.pdf" style="color: #245ED2; text-decoration: none">paper</a>] [<a href="../misc-files/bibs/padmakumar2023extrapolative.txt" style="color: #245ED2; text-decoration: none">bibtex</a>]


<br><br>


<p class="small">
<span style="font-weight:500"> <a name="exactline-acl23-b" href="https://arxiv.org/abs/2211.08714" style="font-size: 12pt; color: #245ED2; text-decoration: none"> Reward Gaming in Conditional Text Generation </a> <br> </span>
<span style="line-height:160%"> Richard Yuanzhe Pang, Vishakh Padmakumar, Thibault Sellam, Ankur P. Parikh, He He </span>
<br>
<span style="line-height:180%; font-size: 10.5pt"> In <i>Proceedings of ACL 2023</i> </span>

<p style="font-size:10pt; margin-left: 15px; line-height:140%"> <b> Abstract:</b> To align conditional text generation model outputs with desired behaviors, there has been an increasing focus on training the model using reinforcement learning (RL) with reward functions learned from human annotations. Under this framework, we identify three common cases where high rewards are incorrectly assigned to undesirable patterns: noise-induced spurious correlation, naturally occurring spurious correlation, and covariate shift. We show that even though learned metrics achieve high performance on the distribution of the data used to train the reward function, the undesirable patterns may be amplified during RL training of the text generation model. While there has been discussion about reward gaming in the RL or safety community, in this discussion piece, we would like to highlight reward gaming in the natural language generation (NLG) community using concrete conditional text generation examples and discuss potential fixes and areas for future work.

<br><br>

[<a href="https://arxiv.org/pdf/2211.08714.pdf" style="color: #245ED2; text-decoration: none">paper</a>] [<a href="https://youtu.be/CdX5do_3geE" style="color: #245ED2; text-decoration: none">15-min talk</a>] [<a href="../misc-files/pang-reward-gaming-slides-acl2023.pdf" style="color: #245ED2; text-decoration: none">slides</a>] [<a href="../misc-files/bibs/pang2023reward.txt" style="color: #245ED2; text-decoration: none">bibtex</a>]


<br><br>


<p class="small">
<span style="font-weight:500"> <a name="exactline-acl23-a" href="https://arxiv.org/abs/2208.12852" style="font-size: 12pt; color: #757575; text-decoration: none"> What Do NLP Researchers Believe? Results of the NLP Community Metasurvey </a> <br> </span>
<span style="line-height:160%"> Julian Michael, Ari Holtzman, Alicia Parrish, Aaron Mueller, Alex Wang, Angelica Chen, Divyam Madaan, Nikita Nangia, Richard Yuanzhe Pang, Jason Phang, Samuel R. Bowman </span>
<br>
<span style="line-height:180%; font-size: 10.5pt"> In <i>Proceedings of ACL 2023</i> </span>

<p style="font-size:10pt; margin-left: 15px; line-height:140%"> <b> Abstract:</b> We present the results of the NLP Community Metasurvey. Run from May to June 2022, the survey elicited opinions on controversial issues, including industry influence in the field, concerns about AGI, and ethics. Our results put concrete numbers to several controversies: For example, respondents are split almost exactly in half on questions about the importance of artificial general intelligence, whether language models understand language, and the necessity of linguistic structure and inductive bias for solving NLP problems. In addition, the survey posed meta-questions, asking respondents to predict the distribution of survey responses. This allows us not only to gain insight on the spectrum of beliefs held by NLP researchers, but also to uncover false sociological beliefs where the community's predictions don't match reality. We find such mismatches on a wide range of issues. Among other results, the community greatly overestimates its own belief in the usefulness of benchmarks and the potential for scaling to solve real-world problems, while underestimating its own belief in the importance of linguistic structure, inductive bias, and interdisciplinary science.

<br><br>

[<a href="https://arxiv.org/pdf/2208.12852.pdf" style="color: #757575; text-decoration: none">paper</a>] [<a href="https://nlpsurvey.net/" style="color: #757575; text-decoration: none">website</a>] [<a href="../misc-files/bibs/michael2023nlp.txt" style="color: #757575; text-decoration: none">bibtex</a>] | by others: [<a href="../misc-files/links/survey-articles.html" style="color: #757575; text-decoration: none">press</a>]


<br><br>


<p class="small">
<span style="line-height:170%"> <u> Publications (2021-2022)</u> — <b>main focus</b>: text generation (learning from rewards, RL), long-document understanding (question answering, summarization) </span>

<p class="small">
<span style="font-weight:500"> <a name="exactline-emnlp22" href="https://arxiv.org/abs/2205.11465" style="font-size: 12pt; color: #22789D; text-decoration: none"> SQuALITY: Building a Long-Document Summarization Dataset the Hard Way </a> <br> </span>
<span style="line-height:160%"> Alex Wang, Richard Yuanzhe Pang, Angelica Chen, Jason Phang, Samuel R. Bowman </span>
<br>
<span style="line-height:180%; font-size: 10.5pt"> In <i>Proceedings of EMNLP 2022</i> </span>

<p style="font-size:10pt; margin-left: 15px; line-height:140%"> <b> Abstract:</b> Summarization datasets are often assembled either by scraping naturally occurring public-domain summaries -- which are nearly always in difficult-to-work-with technical domains -- or by using approximate heuristics to extract them from everyday text -- which frequently yields unfaithful summaries. In this work, we turn to a slower but more straightforward approach to developing summarization benchmark data: We hire highly-qualified contractors to read stories and write original summaries from scratch. To amortize reading time, we collect five summaries per document, with the first giving an overview and the subsequent four addressing specific questions. We use this protocol to collect SQuALITY, a dataset of question-focused summaries built on the same public-domain short stories as the multiple-choice dataset QuALITY (Pang et al., 2021). Experiments with state-of-the-art summarization systems show that our dataset is challenging and that existing automatic evaluation metrics are weak indicators of quality.

<br><br>

[<a href="https://arxiv.org/pdf/2205.11465.pdf" style="color: #22789D; text-decoration: none">paper</a>] [<a href="https://github.com/nyu-mll/SQuALITY/tree/main/data" style="color: #22789D; text-decoration: none">data</a>] [<a href="https://github.com/nyu-mll/SQuALITY" style="color: #22789D; text-decoration: none">code</a>] [<a href="../misc-files/bibs/wang2022squality.txt" style="color: #22789D; text-decoration: none">bibtex</a>] | by others: [<a href="https://www.zero.scrolls-benchmark.com" style="color: #22789D; text-decoration: none">zeroscrolls</a>]


<br><br>


<p class="small">
<span style="font-weight:500"> <a name="exactline-amortized-noisy-channel" href="https://arxiv.org/abs/2112.08670" style="font-size: 12pt; color: #753DA4; text-decoration: none"> Amortized Noisy Channel Neural Machine Translation </a> <br> </span>
<span style="line-height:160%"> Richard Yuanzhe Pang, He He, Kyunghyun Cho </span>
<br>
<span style="line-height:180%; font-size: 10.5pt"> In <i>Proceedings of INLG 2022</i>; best presentation award</span>

<p style="font-size:10pt; margin-left: 15px; line-height:140%"> <b> Abstract:</b> Noisy channel models have been especially effective in neural machine translation (NMT). However, recent approaches like "beam search and rerank" (BSR) incur significant computation overhead during inference, making real-world application infeasible. We aim to study if it is possible to build an amortized noisy channel NMT model such that when we do greedy decoding during inference, the translation accuracy matches that of BSR in terms of reward (based on the source-to-target log probability and the target-to-source log probability) and quality (based on BLEU and BLEURT). We attempt three approaches to train the new model: knowledge distillation, one-step-deviation imitation learning, and Q learning. The first approach obtains the noisy channel signal from a pseudo-corpus, and the latter two approaches aim to optimize toward a noisy-channel MT reward directly. For all three approaches, the generated translations fail to achieve rewards comparable to BSR, but the translation quality approximated by BLEU and BLEURT is similar to the quality of BSR-produced translations. Additionally, all three approaches speed up inference by 1–2 orders of magnitude.

<br><br>

[<a href="https://arxiv.org/pdf/2112.08670.pdf" style="color: #753DA4; text-decoration: none">paper</a>] [<a href="https://www.youtube.com/watch?v=EqpNw7JJvI4" style="color: #753DA4; text-decoration: none">talk</a>] [<a href="../misc-files/pang-amortized-poster-inlg2022.pdf" style="color: #753DA4; text-decoration: none">poster</a>] [<a href="../misc-files/bibs/pang2022amortized.txt" style="color: #753DA4; text-decoration: none">bibtex</a>]


<br><br>


<p class="small">
<span style="font-weight:500"> <a name="exactline-quality" href="https://arxiv.org/abs/2112.08608" style="font-size: 12pt; color: #22789D; text-decoration: none"> QuALITY: Question Answering with Long Input Texts, Yes!</a> <br> </span>
<span style="line-height:160%"> Richard Yuanzhe Pang<sup>*</sup>, Alicia Parrish<sup>*</sup>, Nitish Joshi<sup>*</sup>, Nikita Nangia, Jason Phang, Angelica Chen, Vishakh Padmakumar, Johnny Ma, Jana Thompson, He He, Samuel R. Bowman </span>
<br>
<span style="line-height:180%; font-size: 10.5pt"> In <i>Proceedings of NAACL 2022</i> </span>

<p style="font-size:10pt; margin-left: 15px; line-height:140%"> <b> Abstract:</b> To enable building and testing models on long-document comprehension, we introduce QuALITY, a multiple-choice QA dataset with context passages in English that have an average length of about 5,000 tokens, much longer than typical current models can process. Unlike in prior work with passages, our questions are written and validated by contributors who have read the entire passage, rather than relying on summaries or excerpts. In addition, only half of the questions are answerable by annotators working under tight time constraints, indicating that skimming and simple search are not enough to consistently perform well. Our baseline models perform poorly on this task (55.4%) and significantly lag behind human performance (93.5%). 

<br><br>

[<a href="https://arxiv.org/pdf/2112.08608.pdf" style="color: #22789D; text-decoration: none">paper</a>] [<a href="https://github.com/nyu-mll/quality/tree/main/data" style="color: #22789D; text-decoration: none">data</a>] [<a href="https://github.com/nyu-mll/quality/tree/main/baselines" style="color: #22789D; text-decoration: none">code</a>] [<a href="https://nyu-mll.github.io/quality/" style="color: #22789D; text-decoration: none">leaderboard</a>] [<a href="https://www.youtube.com/watch?v=WAhSW5iP8iw" style="color: #22789D; text-decoration: none">15-min live talk</a>] [<a href="../misc-files/pang-quality-slides-naacl2022.pdf" style="color: #22789D; text-decoration: none">slides</a>] [<a href="../misc-files/bibs/pang2022quality.txt" style="color: #22789D; text-decoration: none">bibtex</a>] | by others: [<a href="https://www.tensorflow.org/datasets/catalog/quality" style="color: #22789D; text-decoration: none">tfds</a>] [<a href="https://www.metaculus.com/questions/9628/question-answering-on-long-texts-by-2025/" style="color: #22789D; text-decoration: none">forecast</a>] [<a href="https://www.science.org/content/article/computers-ace-iq-tests-still-make-dumb-mistakes-can-different-tests-help" style="color: #22789D; text-decoration: none">press mention by <i>Science</i></a>] [<a href="https://www.scrolls-benchmark.com/" style="color: #22789D; text-decoration: none">scrolls</a>] [<a href="https://www.zero.scrolls-benchmark.com" style="color: #22789D; text-decoration: none">zeroscrolls</a>]


<br><br>


<p class="small">
<span style="font-weight:500"> <a name="exactline-acl22" href="https://arxiv.org/abs/2203.13240" style="font-size: 12pt; color: #757575; text-decoration: none">Token Dropping for Efficient BERT Pretraining</a> <br> </span>
<span style="line-height:160%"> Le Hou<sup>*</sup>, Richard Yuanzhe Pang<sup>*</sup>, Tianyi Zhou, Yuexin Wu, Xinying Song, Xiaodan Song, Denny Zhou </span>
<br>
<span style="line-height:180%; font-size: 10.5pt"> In <i>Proceedings of ACL 2022</i> </span>

<p style="font-size:10pt; margin-left: 15px; line-height:140%"> <b> Abstract:</b> Transformer-based models generally allocate the same amount of computation for each token in a given sequence. We develop a simple but effective "token dropping" method to accelerate the pretraining of transformer models, such as BERT, without degrading its performance on downstream tasks. In short, we drop unimportant tokens starting from an intermediate layer in the model to make the model focus on important tokens; the dropped tokens are later picked up by the last layer of the model so that the model still produces full-length sequences. We leverage the already built-in masked language modeling (MLM) loss to identify unimportant tokens with practically no computational overhead. In our experiments, this simple approach reduces the pretraining cost of BERT by 25% while achieving similar overall fine-tuning performance on standard downstream tasks.

<br><br>

[<a href="https://arxiv.org/pdf/2203.13240.pdf" style="color: #757575; text-decoration: none">paper</a>] [<a href="https://github.com/tensorflow/models/tree/master/official/projects/token_dropping" style="color: #757575; text-decoration: none">code</a>] [<a href="https://drive.google.com/file/d/1-j54SpprZvnDGwLKCacF4xs2o0hYpTlL/view?usp=sharing" style="color: #757575; text-decoration: none">talk</a>] [<a href="../misc-files/bibs/hou2022token.txt" style="color: #757575; text-decoration: none">bibtex</a>] | by others: [<a href="../misc-files/links/token-dropping-articles.html" style="color: #757575; text-decoration: none">press</a>] [<a href="https://arxiv.org/abs/2305.15273" style="color: #757575; text-decoration: none">improvement</a>]


<br><br>


<!-- <font size="3"> -->
<p class="small">
<span style="font-weight:500"> <a name="exactline-acl21-b" href="https://arxiv.org/abs/2106.02278" style="font-size: 12pt; color: #245ED2; text-decoration: none"> AgreeSum: Agreement-Oriented Multi-Document Summarization</a> <br> </span>
<span style="line-height:160%"> Richard Yuanzhe Pang<sup>*</sup>, Adam D. Lelkes<sup>*</sup>, Vinh Q. Tran<sup>*</sup>, Cong Yu </span>
<br>
<span style="line-height:180%; font-size: 10.5pt"> In <i>Findings of ACL 2021</i> </span>

<p style="font-size:10pt; margin-left: 15px; line-height:140%"> <b> Abstract:</b> We aim to renew interest in a particular multi-document summarization (MDS) task which we call AgreeSum: agreement-oriented multi-document summarization. Given a cluster of articles, the goal is to provide abstractive summaries that represent information common and faithful to all input articles. Given the lack of existing datasets, we create a dataset for AgreeSum, and provide annotations on article-summary entailment relations for a subset of the clusters in the dataset. We aim to create strong baselines for the task by applying the top-performing pretrained single-document summarization model PEGASUS onto AgreeSum, leveraging both annotated clusters by supervised losses, and unannotated clusters by T5-based entailment-related and language-related losses. Compared to other baselines, both automatic evaluation and human evaluation show better article-summary and cluster-summary entailment in generated summaries. On a separate note, we hope that our article-summary entailment annotations contribute to the community's effort in improving abstractive summarization faithfulness.

<br><br>

[<a href="https://arxiv.org/pdf/2106.02278.pdf" style="color: #245ED2; text-decoration: none">paper</a>] [<a href="https://github.com/google-research-datasets/AgreeSum" style="color: #245ED2; text-decoration: none">data</a>] [<a href="https://drive.google.com/file/d/1EYE3WLxqFpBHeSPU7zQ4gPZsxOE4wj5G/view?usp=sharing" style="color: #245ED2; text-decoration: none">short video</a>] [<a href="../misc-files/bibs/pang2021agreesum.txt" style="color: #245ED2; text-decoration: none">bibtex</a>]


<br><br>


<!-- <font size="3"> -->
<p class="small">
<span style="font-weight:500"> <a name="exactline-acl21-a" href="https://arxiv.org/abs/2106.00840" style="font-size: 12pt; color: #22789D; text-decoration: none"> Comparing Test Sets with Item Response Theory</a> <br> </span>
<span style="line-height:160%"> Clara Vania<sup>*</sup>, Phu Mon Htut<sup>*</sup>, William Huang<sup>*</sup>, Dhara Mungra, Richard Yuanzhe Pang, Jason Phang, Haokun Liu, Kyunghyun Cho, Samuel R. Bowman </span>
<br>
<span style="line-height:180%; font-size: 10.5pt"> In <i>Proceedings of ACL 2021</i> </span>

<p style="font-size:10pt; margin-left: 15px; line-height:140%"> <b> Abstract:</b> Recent years have seen numerous NLP datasets introduced to evaluate the performance of fine-tuned models on natural language understanding tasks. Recent results from large pretrained models, though, show that many of these datasets are largely saturated and unlikely to be able to detect further progress. What kind of datasets are still effective at discriminating among strong models, and what kind of datasets should we expect to be able to detect future improvements? To measure this uniformly across datasets, we draw on Item Response Theory and evaluate 29 datasets using predictions from 18 pretrained Transformer models on individual test examples. We find that Quoref, HellaSwag, and MC-TACO are best suited for distinguishing among state-of-the-art models, while SNLI, MNLI, and CommitmentBank seem to be saturated for current strong models. We also observe span selection task format, which is used for QA datasets like QAMR or SQuAD2.0, is effective in differentiating between strong and weak models.

<br><br>

[<a href="https://arxiv.org/pdf/2106.00840.pdf" style="color: #22789D; text-decoration: none">paper</a>] [<a href="https://github.com/nyu-mll/nlu-test-sets" style="color: #22789D; text-decoration: none">code</a>] [<a href="../misc-files/bibs/vania2021comparing.txt" style="color: #22789D; text-decoration: none">bibtex</a>]


<br><br>


<!-- <font size="3"> -->
<p class="small">
<span style="font-weight:500"> <a name="exactline-iclr21" href="https://arxiv.org/abs/2009.07839" style="font-size: 12pt; color: #245ED2; text-decoration: none"> Text Generation by Learning from Demonstrations</a> <br> </span>
<span style="line-height:160%"> Richard Yuanzhe Pang, He He </span>
<br>
<span style="line-height:180%; font-size: 10.5pt"> In <i>Proceedings of ICLR 2021</i> </span>

<p style="font-size:10pt; margin-left: 15px; line-height:140%"> <b> Abstract:</b> Current approaches to text generation largely rely on autoregressive models and maximum likelihood estimation. This paradigm leads to (i) diverse but low-quality samples due to mismatched learning objective and evaluation metric (likelihood vs. quality) and (ii) exposure bias due to mismatched history distributions (gold vs. model-generated). To alleviate these problems, we frame text generation as an offline reinforcement learning (RL) problem with expert demonstrations (i.e., the reference), where the goal is to maximize quality given model-generated histories. We propose GOLD (generation by off-policy learning from demonstrations): an easy-to-optimize algorithm that learns from the demonstrations by importance weighting. Intuitively, GOLD upweights confident tokens and downweights unconfident ones in the reference during training, avoiding optimization issues faced by prior RL approaches that rely on online data collection. According to both automatic and human evaluation, models trained by GOLD outperform those trained by MLE and policy gradient on summarization, question generation, and machine translation. Further, our models are less sensitive to decoding algorithms and alleviate exposure bias.

<br><br>

[<a href="../misc-files/pang+he-gold-paper.pdf" style="color: #245ED2; text-decoration: none">paper</a>] [<a href="https://openreview.net/forum?id=RovX-uQ1Hua" style="color: #245ED2; text-decoration: none">openreview</a>] [<a href="../misc-files/pang+he-gold-poster.pdf" style="color: #245ED2; text-decoration: none">poster</a>] [<a href="../misc-files/pang+he-gold-slides.pdf" style="color: #245ED2; text-decoration: none">slides</a>] [<a href="
https://github.com/yzpang/gold-off-policy-text-gen-iclr21" style="color: #245ED2; text-decoration: none">code</a>] [<a href="../misc-files/pang-gold-discussion-2022-06.pdf" style="color: #245ED2; text-decoration: none">discussion</a>] [<a href="../misc-files/bibs/pang2021text.txt" style="color: #245ED2; text-decoration: none">bibtex</a>] | by others: [<a href="https://iclr-blog-track.github.io/2022/03/25/text-gen-via-lfd/" style="color: #245ED2; text-decoration: none">ICLR blog by other authors</a>] [<a href="https://www.science.org/stoken/author-tokens/ST-905/full" style="color: #245ED2; text-decoration: none">GOLD in AlphaCode, <i>Science</i></a>] [<a href="https://storage.googleapis.com/deepmind-media/AlphaCode2/AlphaCode2_Tech_Report.pdf" style="color: #245ED2; text-decoration: none">GOLD as the main learning objective in AlphaCode 2, Dec 2023</a>]


<br><br>


<p class="small">
<span style="line-height:170%"> <u> Publications (2019-2020)</u> — <b>main focus</b>: text generation (textual style transfer, non-autoregressive translation, decoding), energy-based network in NLP </span>


<p class="small">
<span style="font-weight:500"> <a name="exactline-spnlp20" href="https://arxiv.org/abs/1911.02891" style="font-size: 12pt; color: #245ED2; text-decoration: none"> Improving Joint Training of Inference Networks and Structured Prediction Energy Networks </a> <br> </span>
<span style="line-height:160%"> Lifu Tu, Richard Yuanzhe Pang, Kevin Gimpel </span>
<br>
<span style="line-height:180%; font-size: 10.5pt"> In <i>Proceedings of EMNLP 2020 Workshop on Structured Prediction for NLP (SPNLP)</i>; spotlight paper </span>

<p style="font-size:10pt; margin-left: 15px; line-height:140%"> <b> Abstract:</b> Deep energy-based models are powerful, but pose challenges for learning and inference (Belanger and McCallum, 2016). Tu and Gimpel (2018) developed an efficient framework for energy-based models by training "inference networks" to approximate structured inference instead of using gradient descent. However, their alternating optimization approach suffers from instabilities during training, requiring additional loss terms and careful hyperparameter tuning. In this paper, we contribute several strategies to stabilize and improve this joint training of energy functions and inference networks for structured prediction. We design a compound objective to jointly train both cost-augmented and test-time inference networks along with the energy function. We propose joint parameterizations for the inference networks that encourage them to capture complementary functionality during learning. We empirically validate our strategies on two sequence labeling tasks, showing easier paths to strong performance than prior work, as well as further improvements with global energy terms.

<br><br>

[<a href="https://arxiv.org/pdf/1911.02891.pdf" style="color: #245ED2; text-decoration: none">paper</a>] [<a href="../misc-files/pang-spen-infnet-joint-training-slides.pdf" style="color: #245ED2; text-decoration: none">my slides</a>] [<a href="../misc-files/bibs/tu2019improving.txt" style="color: #245ED2; text-decoration: none">bibtex</a>] 


<br><br>


<!-- <font size="3"> -->
<p class="small">
<span style="font-weight:500"> <a name="exactline-emnlp20-a" href="https://arxiv.org/abs/2002.02492" style="font-size: 12pt; color: #245ED2; text-decoration: none"> Consistency of a Recurrent Language Model With Respect to Incomplete Decoding </a> <br> </span>
<span style="line-height:160%"> Sean Welleck, Ilia Kulikov, Jaedeok Kim, Richard Yuanzhe Pang, Kyunghyun Cho </span>
<br>
<span style="line-height:180%; font-size: 10.5pt"> In <i>Proceedings of EMNLP 2020</i>; also appearing in the non-archival <i>DeepMath 2020</i> </span>

<p style="font-size:10pt; margin-left: 15px; line-height:140%"> <b> Abstract:</b> Despite strong performance on a variety of tasks, neural sequence models trained with maximum likelihood have been shown to exhibit issues such as length bias and degenerate repetition. We study the related issue of receiving infinite-length sequences from a recurrent language model when using common decoding algorithms. To analyze this issue, we first define inconsistency of a decoding algorithm, meaning that the algorithm can yield an infinite-length sequence that has zero probability under the model. We prove that commonly used incomplete decoding algorithms -- greedy search, beam search, top-k sampling, and nucleus sampling -- are inconsistent, despite the fact that recurrent language models are trained to produce sequences of finite length. Based on these insights, we propose two remedies which address inconsistency: consistent variants of top-k and nucleus sampling, and a self-terminating recurrent language model. Empirical results show that inconsistency occurs in practice, and that the proposed methods prevent inconsistency.

<br><br>

[<a href="https://arxiv.org/pdf/2002.02492.pdf" style="color: #245ED2; text-decoration: none">paper</a>] [<a href="https://github.com/uralik/consistency-lm" style="color: #245ED2; text-decoration: none">code</a>] [<a href="../misc-files/bibs/welleck2020consistency.txt" style="color: #245ED2; text-decoration: none">bibtex</a>]


<br><br>


<p class="small">
<span style="font-weight:500"> <a href="https://arxiv.org/abs/2005.00850" name="exactline-acl20-a" style="font-size: 12pt; color: #753DA4; text-decoration: none"> ENGINE: Energy-Based Inference Networks for Non-Autoregressive Machine Translation </a> <br> </span>
<span style="line-height:160%"> Lifu Tu, Richard Yuanzhe Pang, Sam Wiseman, Kevin Gimpel</span>
<br>
<span style="line-height:180%; font-size: 10.5pt"> In <i>Proceedings of ACL 2020</i> </span>

<p style="font-size:10pt; margin-left: 15px; line-height:140%"> <b> Abstract:</b> We propose to train a non-autoregressive machine translation model to minimize the energy defined by a pretrained autoregressive model. In particular, we view our non-autoregressive translation system as an inference network (Tu and Gimpel, 2018) trained to minimize the autoregressive teacher energy. This contrasts with the popular approach of training a non-autoregressive model on a distilled corpus consisting of the beam-searched outputs of such a teacher model. Our approach, which we call ENGINE (ENerGy-based Inference NEtworks), achieves state-of-the-art non-autoregressive results on the IWSLT 2014 DE-EN and WMT 2016 RO-EN datasets, approaching the performance of autoregressive models.

<br><br>

[<a href="https://arxiv.org/pdf/2005.00850.pdf" style="color: #753DA4; text-decoration: none">paper</a>] [<a href="https://github.com/lifu-tu/ENGINE" style="color: #753DA4; text-decoration: none">code</a>] [<a href="../misc-files/bibs/tu2020engine.txt" style="color: #753DA4; text-decoration: none">bibtex</a>]


<br><br>


<p class="small">
<span style="font-weight:500"> <a href="https://arxiv.org/abs/2005.00628" name="exactline-acl20-b" style="font-size: 12pt; color: #22789D; text-decoration: none"> Intermediate-Task Transfer Learning with Pretrained Language Models: When and Why Does It Work? </a> <br> </span>
<span style="line-height:160%"> Yada Pruksachatkun, Jason Phang, Haokun Liu, Phu Mon Htut, Xiaoyi Zhang, Richard Yuanzhe Pang, Clara Vania, Katharina Kann, Samuel R. Bowman</span>
<br>
<span style="line-height:180%; font-size: 10.5pt"> In <i>Proceedings of ACL 2020</i> </span>

<p style="font-size:10pt; margin-left: 15px; line-height:140%"> <b> Abstract:</b> While pretrained models such as BERT have shown large gains across natural language understanding tasks, their performance can be improved by further training the model on a data-rich intermediate task, before fine-tuning it on a target task. However, it is still poorly understood when and why intermediate-task training is beneficial for a given target task. To investigate this, we perform a large-scale study on the pretrained RoBERTa model with 110 intermediate-target task combinations. We further evaluate all trained models with 25 probing tasks meant to reveal the specific skills that drive transfer. We observe that intermediate tasks requiring high-level inference and reasoning abilities tend to work best. We also observe that target task performance is strongly correlated with higher-level abilities such as coreference resolution. However, we fail to observe more granular correlations between probing and target task performance, highlighting the need for further work on broad-coverage probing benchmarks. We also observe evidence that the forgetting of knowledge learned during pretraining may limit our analysis, highlighting the need for further work on transfer learning methods in these settings.

<br><br>

[<a href="https://arxiv.org/pdf/2005.00628.pdf" style="color: #22789D; text-decoration: none">paper</a>] [<a href="../misc-files/bibs/pruksachatkun2020intermediate.txt" style="color: #22789D; text-decoration: none">bibtex</a>]


<br><br>


<p class="small">
<span style="font-weight:500"> <a name="exactline-wngt19-a" href="https://arxiv.org/abs/1810.11878" style="font-size: 12pt; color: #245ED2; text-decoration: none"> Unsupervised Evaluation Metrics and Learning Criteria for Non-Parallel Textual Transfer </a> <br> </span>
<span style="line-height:160%"> Richard Yuanzhe Pang, Kevin Gimpel </span>
<br>
<span style="line-height:180%; font-size: 10.5pt"> In <i>Proceedings of EMNLP 2019 Workshop on Neural Generation and Translation (WNGT)</i></span>

<p style="font-size:10pt; margin-left: 15px; line-height:140%"> <b> Abstract:</b> We consider the problem of automatically generating textual paraphrases with modified attributes or properties, focusing on the setting without parallel data (Hu et al., 2017; Shen et al., 2017). This setting poses challenges for evaluation. We show that the metric of post-transfer classification accuracy is insufficient on its own, and propose additional metrics based on semantic preservation and fluency as well as a way to combine them into a single overall score. We contribute new loss functions and training strategies to address the different metrics. Semantic preservation is addressed by adding a cyclic consistency loss and a loss based on paraphrase pairs, while fluency is improved by integrating losses based on style-specific language models. We experiment with a Yelp sentiment dataset and a new literature dataset that we propose, using multiple models that extend prior work (Shen et al., 2017). We demonstrate that our metrics correlate well with human judgments, at both the sentence-level and system-level. Automatic and manual evaluation also show large improvements over the baseline method of Shen et al. (2017). We hope that our proposed metrics can speed up system development for new textual transfer tasks while also encouraging the community to address our three complementary aspects of transfer quality.

<br><br>

[<a href="https://arxiv.org/pdf/1810.11878.pdf" style="color: #245ED2; text-decoration: none">paper</a>] [<a href="https://arxiv.org/pdf/1810.11878.pdf#page=11" style="color: #245ED2; text-decoration: none">supplementals</a>] [<a href="../misc-files/pang+gimpel-textual-transfer-poster.pdf" style="color: #245ED2; text-decoration: none">poster</a>] [<a href="../misc-files/bibs/pang2018unsupervised.txt" style="color: #245ED2; text-decoration: none">bibtex</a>]


<br><br>


<p class="small">
<span style="font-weight:500"> <a name="exactline-wngt19-b" href="https://arxiv.org/abs/1910.03747" style="font-size: 12pt; color: #245ED2; text-decoration: none"> The Daunting Task of Real-World Textual Style Transfer Auto-Evaluation </a> <br> </span>
<span style="line-height:180%"> Richard Yuanzhe Pang </span>
<br>
<span style="line-height:140%; font-size: 10.5pt"> Extended abstract in <i>EMNLP 2019 Workshop on Neural Generation and Translation (WNGT)</i>; abstract in <i>Proceedings of the Workshop on Noisy User-generated Text (W-NUT)</i></span> 

<p style="font-size:10pt; margin-left: 15px; line-height:140%"> <b> Abstract:</b> The difficulty of textual style transfer lies in the lack of parallel corpora. Numerous advances have been proposed for the unsupervised generation. However, significant problems remain with the auto-evaluation of style transfer tasks. Based on the summary of Pang and Gimpel (2018) and Mir et al. (2019), style transfer evaluations rely on three criteria: style accuracy of transferred sentences, content similarity between original and transferred sentences, and fluency of transferred sentences. We elucidate the problematic current state of style transfer research. Given that current tasks do not represent real use cases of style transfer, current auto-evaluation approach is flawed. This discussion aims to bring researchers to think about the future of style transfer and style transfer evaluation research.

<br><br>

[<a href="https://arxiv.org/pdf/1910.03747.pdf" style="color: #245ED2; text-decoration: none">paper</a>] [<a href="../misc-files/pang-textual-transfer-problem-poster.pdf" style="color: #245ED2; text-decoration: none">poster</a>] [<a href="../misc-files/bibs/pang2019daunting.txt" style="color: #245ED2; text-decoration: none">bibtex</a>]
</p>












