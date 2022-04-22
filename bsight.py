
"""
- my understanding is that i can take the boxes output and compare with the boxes values given by the data.
- if +- some amount punya distance then ok? - ok turns out this is IOU 
-  Final report says IOU not goof enough, need to know if the symbol is detected correctly so that means 
   we'll need to get the word from the back and compare to the topic we detected and see how similar they are? WordEmbedModel??

- Suggested Evaluation metrics after getting the IOU + WordEmbedModel to see how similar
    - Confusion matrix
    - f1_score
    - cohen_kappa score

- but they have coco_eval and evaluate???

Seniors Suggested
- add SentenceEmbreddeinModel to be able to handle Q&A + Slogans
- k-clustering for labels 
"""
