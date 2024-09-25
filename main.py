import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt

file_path = 'bot_responses.csv'
df = pd.read_csv(file_path)


def calculate_cosine_similarity(df, target_col, real_col):
    vectorizer = TfidfVectorizer()

    combined_text = df[target_col].astype(str).tolist() + df[real_col].astype(str).tolist()
    tfidf_matrix = vectorizer.fit_transform(combined_text)

    target_matrix = tfidf_matrix[:len(df)]
    real_matrix = tfidf_matrix[len(df):]

    cos_sim = cosine_similarity(target_matrix, real_matrix).diagonal()

    return cos_sim

def calculate_rouge_recall(df, target_col, real_col):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    rouge_recall = []

    for target, real in zip(df[target_col].astype(str), df[real_col].astype(str)):
        scores = scorer.score(target, real)
        rouge_recall.append(scores['rouge1'].recall)

    return rouge_recall


def calculate_bleu_recall(df, target_col, real_col):
    bleu_recall = []
    smooth_fn = SmoothingFunction().method1

    for target, real in zip(df[target_col].astype(str), df[real_col].astype(str)):
        reference = [target.split()]
        candidate = real.split()
        score = sentence_bleu(reference, candidate, smoothing_function=smooth_fn)
        bleu_recall.append(score)

    return bleu_recall


response_pairs = [
    ('Target_response_bella', 'Bella_w4'),
    ('Target_response_mia', 'Mia_w3'),
    ('Target_response_mike', 'Mike_w1'),
    ('Target_response_olivia', 'Olivia_w2')
]


cosine_similarities = {}
rouge_recalls = {}
bleu_recalls = {}


for target_col, real_col in response_pairs:
    cosine_sim = calculate_cosine_similarity(df, target_col, real_col)
    rouge_recall = calculate_rouge_recall(df, target_col, real_col)
    bleu_recall = calculate_bleu_recall(df, target_col, real_col)


    cosine_similarities[target_col] = cosine_sim
    rouge_recalls[target_col] = rouge_recall
    bleu_recalls[target_col] = bleu_recall


cosine_df = pd.DataFrame(cosine_similarities)
rouge_df = pd.DataFrame(rouge_recalls)
bleu_df = pd.DataFrame(bleu_recalls)


fig, axs = plt.subplots(2, 2, figsize=(12, 10))


bot_names = ['Bella', 'Mia', 'Mike', 'Olivia']
target_columns = ['Target_response_bella', 'Target_response_mia', 'Target_response_mike', 'Target_response_olivia']


for i, ax in enumerate(axs.flat):
    bot_name = bot_names[i]
    target_col = target_columns[i]

    ax.plot(cosine_df.index, cosine_df[target_col], label='Cosine Similarity', color='blue')

    ax.plot(rouge_df.index, rouge_df[target_col], label='ROUGE Recall', color='red')

    ax.plot(bleu_df.index, bleu_df[target_col], label='BLEU Recall', color='green')

    ax.set_title(f'{bot_name} - Cosine, ROUGE & BLEU Recalls')
    ax.set_xlabel('Data Point Index')
    ax.set_ylabel('Score')
    ax.legend()

plt.tight_layout()
plt.show()

for bot, target_col in zip(bot_names, target_columns):
    avg_cosine = np.mean(cosine_df[target_col])
    avg_rouge = np.mean(rouge_df[target_col])
    avg_bleu = np.mean(bleu_df[target_col])
    print(f"Bot: {bot}")
    print(f"Average Cosine Similarity: {avg_cosine:.4f}")
    print(f"Average ROUGE Recall: {avg_rouge:.4f}")
    print(f"Average BLEU Recall: {avg_bleu:.4f}\n")
