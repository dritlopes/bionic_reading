import re
import pandas as pd
import spacy
from collections import defaultdict
from spacy.cli import download
from spacy_syllables import SpacySyllables

from compute_saliency import calculate_saliency_values


def create_text_and_question_files(chatgpt_output_filepath):

    text_col, title_col, question_col, answer_col, correct_answer_col = [], [], [], [], []

    with open(chatgpt_output_filepath) as infile:

        file = infile.read()

    parts = re.split('---', file)

    # for each text + question
    for part in parts:

        part = part.strip()

        if part.startswith('**'):

            # text, question, and answer
            sub_parts = re.split('\n\n', part)

            for sub_part in sub_parts:

                # sort question
                if '**Vraag**' in sub_part:

                    question = sub_part.replace('**Vraag**:', '')
                    lines = question.split('\n')
                    answers = ''

                    for line in lines:

                        # sort answers
                        if '**Antwoord**' in line:
                            line = line.replace('**Antwoord**: ', '')
                            correct = line[0]
                            correct_answer_col.append(correct)

                        elif not line.startswith('a)') and not line.startswith('b)'):
                            question = line.strip()
                            question_col.append(question)

                        else:
                            answers = answers + ' ' + line
                            answers = answers.strip()
                    answer_col.append(answers)

                # sort text
                else:
                    text_title = re.match(r'\*+\d.*\*+', sub_part).group(0)
                    title_col.append(re.sub(r'^\*+\d+\.\s*|\*+$', '', text_title))
                    text = sub_part.replace(text_title, '')
                    text = text.strip()
                    text_col.append(text)

    df_texts = pd.DataFrame({'trial_id': [i for i in range(len(text_col))],
                             'title': title_col,
                            'text': text_col})
    df_questions = pd.DataFrame({'trial_id': [i for i in range(len(text_col))],
                                'question': question_col,
                                'answers': answer_col,
                                'correct_answer': correct_answer_col})

    return df_texts, df_questions

def extract_syllables(token):

    if token._.syllables:
        syllables = token._.syllables
        # correct 'over' as syllable in Dutch
        if token.text.lower().startswith('over') and 'over' in token._.syllables:
            syllables = []
            for s in token._.syllables:
                if s == 'over':
                    syllables.extend(['o', 'ver'])
                else:
                    syllables.append(s)
        if token.text[0].isupper():
            syllables[0] = syllables[0].replace(syllables[0][0], syllables[0][0].upper())
    else:
        syllables = None

    return syllables

def create_word_file(texts_df, nlp, syllabify=True, pos_tag=True, length=True):

    if syllabify:
        nlp.add_pipe("syllables", after="tagger")

    df_dict = defaultdict(list)

    for text_id, text in zip(texts_df['trial_id'].tolist(),texts_df['text'].tolist()):
        doc = nlp(text)
        word_id = 0
        for i, token in enumerate(doc):
            df_dict['trial_id'].append(text_id)
            df_dict['text'].append(text)
            df_dict['word_id'].append(word_id)
            df_dict['word'].append(token.text)
            word_id += 1
            if syllabify:
                df_dict['syllable'].append(extract_syllables(token))
            if pos_tag:
                df_dict['pos_tag'].append(token.pos_)
            if length:
                df_dict['length'].append(len(token.text))

    df = pd.DataFrame(df_dict)

    return df

def main():

    # # Sort texts and questions from chatgpt output into a csv file
    chatgpt_texts_filepath = "data/chatgpt_texts.txt"
    texts_filepath = "data/texts.csv"
    questions_filepath = "data/questions.csv"
    texts_df, questions_df = create_text_and_question_files(chatgpt_texts_filepath)
    texts_df.to_csv(texts_filepath, index=False)
    questions_df.to_csv(questions_filepath, index=False)

    # Create csv with each word as a row (word data)
    spacy_model_name = "nl_core_news_sm"
    words_filepath = "data/words.csv"
    syllabify, pos_tag, length = True, True, True
    # Try loading the model. If not found, download and load it
    try:
        nlp = spacy.load(spacy_model_name)
    except OSError:
        print(f"Model '{spacy_model_name}' not found. Downloading...")
        download(spacy_model_name)
        nlp = spacy.load(spacy_model_name)
    words_df = create_word_file(texts_df, nlp, syllabify, pos_tag, length)
    words_df.to_csv(words_filepath)

    # Extract saliency values and added them as a column to word data
    language_model_name = "GroNLP/bert-base-dutch-cased" # "GroNLP/gpt2-small-dutch"
    saliency_path = f'data/{language_model_name.replace("/","_")}_saliency.csv'
    final_data_path = f'data/full_data_{language_model_name}.csv'
    texts_df = pd.read_csv('data/texts.csv')
    words_df = pd.read_csv('data/words.csv')
    saliency_df, words_plus_saliency_df = calculate_saliency_values(texts_df, words_df, language_model_name, saliency_path)
    saliency_df.to_csv(saliency_path, index=False)
    words_plus_saliency_df.to_csv(final_data_path, index=False)

if __name__ == "__main__":
    main()
