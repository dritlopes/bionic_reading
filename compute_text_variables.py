import re
import pandas as pd
import spacy
from collections import defaultdict
import math
from spacy.cli import download
from spacy_syllables import SpacySyllables
from compute_saliency import calculate_saliency_values, processing_to_align_with_opensesame


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

    # replace generated texts which have language mistakes (corrected by RA Sophie who is also native Dutch)
    map_texts = {6: "Bij het opruimen van haar oma’s huis vond Elise een oud dagboek. De eerste bladzijden waren gevuld met jeugdherinneringen, maar naarmate ze verder las, ontdekte ze geheimen die haar oma nooit had verteld. Elise kwam erachter dat haar oma ooit een grote liefde had gehad, een man met wie ze nooit had kunnen trouwen. Dit verhaal had haar oma altijd met zich meegedragen, zonder het ooit met iemand te delen. Het dagboek gaf Elise een nieuw perspectief op de vrouw die ze dacht zo goed te kennen.",
                 11: "David wandelde in de vroege ochtend door het bos toen de mist ineens dikker werd. De paden om hem heen vervaagden en hij hoorde niets behalve zijn eigen ademhaling. Hij stond stil, luisterde naar de stilte en voelde een lichte angst opkomen. Maar toen zag hij in de verte een vage schaduw van een figuur. De schaduw bewoog langzaam, alsof deze ook verdwaald was. David twijfelde of hij dichterbij moest gaan of terug naar huis moest keren. Uiteindelijk besloot hij de figuur te volgen; zijn nieuwsgierigheid won het van zijn angst.",
                 13: "Jaarlijks werd er in het dorp een festival gehouden om de oogst te vieren, maar dit jaar leek iedereen het vergeten te zijn. Thomas liep door de straten en zag geen versieringen, geen kramen, niets dat op een feestdag leek. Hij vroeg de oude dorpsbewoner wat er gebeurd was. De man glimlachte mysterieus en zei dat sommige tradities alleen doorgaan als mensen er echt in geloven. Thomas besloot dat hij die avond toch het festival zou vieren, al was het alleen. Toen de zon onderging, stak hij een kaars aan op het dorpsplein.",
                 16: "Mila droeg altijd de ring die ze van haar oma had geërfd. Op een dag merkte ze dat de ring weg was. Ze zocht het hele huis door, maar hij was nergens te vinden. Ze herinnerde zich dat ze hem tijdens een wandeling langs het meer had gedragen. In paniek ging ze terug naar het pad langs het water. Na uren zoeken in de modder gaf ze het op. Terwijl ze weg wilde lopen, blonk er iets in het gras op. De ring lag daar, alsof hij op haar had gewacht.",
                 17: "De oude bibliotheek stond bekend om zijn mysterieuze verleden. Mensen vertelden verhalen over een geest die 's avonds door de gangen zou dwalen. Marit, een sceptische studente, besloot een nacht in de bibliotheek door te brengen om te bewijzen dat het slechts verzinsels waren. Die nacht hoorde ze zachte voetstappen en geritsel van papieren. Omdat ze wist dat niemand anders in de bibliotheek was, bleef ze rustig. Maar toen een stapel boeken opeens van een tafel viel, voelde ze een koude rilling over haar rug. Misschien was er toch meer aan de hand.",
                 19: "Toen Mark voor het eerst zijn nieuwe buurman zag, was hij meteen op zijn hoede. De man leek afstandelijk en sprak nauwelijks met iemand in de buurt. Mark besloot dat er iets vreemds aan de hand was en begon de buurman in de gaten te houden. Elke avond om precies acht uur verliet de man zijn huis en keerde hij pas uren later terug. Op een dag besloot Mark hem te volgen, nieuwsgierig naar wat de man elke avond deed. Wat hij ontdekte, was echter totaal onverwacht: de buurman werkte als vrijwilliger in een opvanghuis.",
                 22: "In een hoek van de bibliotheek vond Tim een oud boek, bedekt met stof. De titel was vaag leesbaar, maar het trok zijn aandacht. Toen hij het opensloeg, ontdekte hij dat het vol stond met verhalen van vergeten helden, mensen die door de geschiedenis waren vergeten. Terwijl hij las, werd hij steeds meer in het boek gezogen, alsof de verhalen tot leven kwamen. Tim besloot het boek mee naar huis te nemen en elke nacht een verhaal te lezen. Het voelde alsof hij iets kostbaars in handen had.",
                 32: "Anna vond een oude brief tussen de pagina’s van een boek dat al jaren in haar kast stond. De brief was geschreven door een verre tante die ze nooit had gekend. In de brief stond een verhaal over verborgen familiegoud, verstopt op een geheime plek. Anna vond het vreemd dat niemand in de familie hier ooit over had gesproken. Ze besloot de aanwijzingen in de brief te volgen, in de hoop dat het geheim nog altijd ergens verborgen lag.",
                 34: "Jaren geleden hadden Tom en zijn vriend Sam elkaar een belofte gedaan. Ze zouden samen een grote reis maken als ze volwassen waren. Maar het leven had hen allebei in verschillende richtingen gestuurd. Op een dag kreeg Tom een bericht van Sam, die voorstelde om eindelijk hun belofte waar te maken. Tom twijfelde; hij had nu een baan en verantwoordelijkheden. Maar de herinnering aan hun belofte was sterk, en Tom wist dat hij deze kans moest grijpen. Ze zouden eindelijk hun droomreis maken.",
                 54: "Elke nacht, precies om middernacht, hoorde Julia een vreemd geluid vanuit de kelder van haar huis. Het klonk alsof iets of iemand op de muren tikte. Ze was er altijd te bang voor geweest om te gaan kijken, maar op een avond verzamelde ze haar moed en ging ze naar beneden. Met een zaklamp in haar hand opende ze de kelderdeur. Tot haar verbazing vond ze niets ongewoons, behalve een oude klok die stilletjes tegen de muur tikte.",
                 55: "Toen de zomer aanbrak, besloot Eva om een lange reis te maken. Ze had altijd al gedroomd van het verkennen van verre landen, en nu had ze eindelijk de kans. Met een rugzak vol benodigdheden en een kaart in haar hand, begon ze aan haar avontuur. Haar eerste stop was een klein dorpje aan de kust, waar ze lokale vissers ontmoette en verhalen uitwisselde over hun levens. Eva besefte dat deze reis haar niet alleen naar nieuwe plaatsen zou brengen, maar ook naar nieuwe inzichten over zichzelf.",
                 59: "Tijdens het opruimen van de zolder vond Emma een brief die achter een kast was gevallen. De brief was oud en de inkt was vervaagd, maar ze kon nog net de naam van haar moeder lezen op de envelop. Nieuwsgierig opende ze de brief en las ze woorden vol liefde en hoop, geschreven door iemand die ze niet kende. Emma besefte dat deze brief waarschijnlijk nooit bedoeld was om gevonden te worden, maar toch voelde ze zich erdoor geraakt.",
                 62: "Liesbeth erfde een groot huis van haar tante, die ze nauwelijks kende. In de gang hing een enorm schilderij van een vrouw die eruitzag als een verre voorouder. Elke keer als Liesbeth langs het schilderij liep, leek het alsof de ogen van de vrouw haar volgden. Het maakte haar ongemakkelijk, maar ze kon het schilderij niet weghalen, omdat het een erfstuk was. Op een avond, toen ze het schilderij van dichterbij bekeek, ontdekte ze een kleine inscriptie in de hoek, een datum die verwees naar een mysterieus familiegeheim.",
                 65: 'Op een koude winteravond liep Thomas door de stad toen een vreemde hem aansprak. De man droeg een lange jas en een hoed die zijn gezicht gedeeltelijk verborg. Hij vroeg Thomas om de weg naar een straat waar Thomas nog nooit van had gehoord. Hoewel Thomas zijn best deed om de man te helpen, leek de man niet echt geïnteresseerd in zijn uitleg. Hij glimlachte alleen en zei: "Je hebt me al meer geholpen dan je denkt." Daarna liep hij weg, de mist in, en liet Thomas verbaasd achter.',
                 73: "In het antiekwinkeltje vond Mila een oude spiegel met prachtige gravures. Toen ze de spiegel omdraaide, ontdekte ze iets vreemds: op de achterkant stond een reeks symbolen gekerfd. Nieuwsgierig ging ze op onderzoek uit en ontdekte ze dat de symbolen een geheime boodschap vormden, maar ze wist niet wat het betekende. Ze besloot de spiegel mee naar huis te nemen en de boodschap verder te ontrafelen. Het voelde alsof de spiegel haar naar een verborgen geschiedenis leidde.",
                 85: "Na jaren van sparen ging Jan eindelijk op zijn droomreis naar Japan. Hij had altijd al gefascineerd gekeken naar de cultuur en geschiedenis van het land. Zijn eerste stop was Tokio, waar de felle lichten en drukte hem overweldigden. Toch vond hij ook momenten van rust, vooral toen hij de serene tempels bezocht. De contrasten tussen traditie en moderniteit maakten de reis voor hem bijzonder. Jan wist dat hij deze ervaring nooit zou vergeten en hij maakte al plannen om terug te keren.",
                 96: "Sofie bezocht regelmatig de kunstgalerie in de stad, maar één schilderij trok steeds haar aandacht. Het was een eenvoudig landschap, maar er was iets aan de kleuren en de rust die het uitstraalde dat haar telkens weer naar het schilderij deed terugkeren. Na weken van twijfelen besloot ze het te kopen. Toen ze het schilderij ophing in haar woonkamer, vulde het de kamer met een bijzondere sfeer. Ze wist dat ze de juiste keuze had gemaakt."}
    corrected_texts = []
    for text_id, text in zip(df_texts['trial_id'].tolist(), df_texts['text'].tolist()):
        if text_id in map_texts.keys():
            new_text = map_texts[text_id]
        else:
            new_text = text
        # make quotations consistent (curly to straight) to make sure tokenization later is consistent
        new_text = new_text.replace('’', '\'')
        corrected_texts.append(new_text)
    df_texts['text'] = corrected_texts

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

def add_missing_syllables(df):

    map_syllables_missing = {
        'oma\'s': ['o', 'ma\'s'],
        '\'s': ['\'s'],
        'hand': ['hand'],
        'Isa\'s': ['I', 'sa\'s'],
        'Sara\'s': ['Sa', 'ra\'s'],
        'zo\'n': ['zo\'n'],
        'pagina\'s': ['pa', 'gi', 'na\'s'],
        'foto\'s': ['fo', 'to\'s'],
        'echo\'s': ['e', 'cho\'s'],
        '10': ['10'],
        'opa\'s': ['o', 'pa\'s'],
        'auto\'s': ['au', 'to\'s']
    }
    syllables = []
    for text_id, word_id, word, syllable in zip(df['trial_id'], df['word_id'], df['word'], df['syllable']):
        if syllable is None:
            if word in map_syllables_missing.keys():
                syllables.append(map_syllables_missing[word])
            else:
                syllables.append([])
        else:
            syllables.append(syllable)
    df['syllable'] = syllables

    return df

def get_letter_segments(word):

    # Split: start, pvl_segment (to be bolded), and end.

    word_len = len(word)
    # floor division
    half_index = word_len // 2 - 1

    if word_len == 3:  # Three-letter words should have the middle letter bolded
        start_segment, pvl_segment, end_segment = word[0], word[1], word[2]

    elif word_len < 9:  # Short words
        # Letter to the left of center
        start_segment = word[:half_index]
        pvl_segment = word[half_index]
        end_segment = word[half_index + 1:]

    else:
        if word_len % 2 != 0: half_index = word_len // 2
        start_segment = word[:half_index - 1]
        pvl_segment = word[half_index - 1: half_index + 1]
        end_segment = word[half_index + 1:]

    return [start_segment, pvl_segment, end_segment]

def create_word_file(texts_df, nlp, syllabify=True, pos_tag=True, length=True, pvl=True):

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
            if pvl:
                df_dict['pvl'].append(get_letter_segments(token.text))

    df = pd.DataFrame(df_dict)

    return df

def test_alignment_opensesame_words(spacy_df, os_df):

    # OS file only has experimental texts (ids 20 to 100)
    spacy_df = spacy_df[(spacy_df['trial_id'] < 100) & (spacy_df['trial_id'] > 19)]
    for spacy_text_id, spacy_text_rows in spacy_df.groupby('trial_id'):
        os_text_rows = os_df[os_df['paragraph'] == spacy_text_id]
        for spacy_word_id, spacy_word in zip(spacy_text_rows['word_id'].tolist(), spacy_text_rows['word'].tolist()):
            os_word_id = os_text_rows[os_text_rows['word_index']==spacy_word_id]['word_index'].tolist()
            os_word = os_text_rows[os_text_rows['word_name']==spacy_word]['word_name'].tolist()
            if len(os_word_id) > 0 and len(os_word) > 0:
                os_word_id = os_word_id[0]
                os_word = os_word[0]
                if (spacy_word_id, spacy_word) != (os_word_id, os_word):
                    print(f'Spacy word [{spacy_text_id}, {spacy_word_id}, {spacy_word}] not matching OS word [{os_word_id}, {os_word}]')
                    print()
            else:
                print(f'Not able to find [{spacy_text_id}, {spacy_word_id}, {spacy_word}] in OS list')
                print()

def main():

    # Sort texts and questions from chatgpt output into a csv file
    chatgpt_texts_filepath = "data/chatgpt_texts.txt"
    texts_filepath = "data/texts.csv"
    questions_filepath = "data/questions.csv"
    texts_df, questions_df = create_text_and_question_files(chatgpt_texts_filepath)
    texts_df.to_csv(texts_filepath, index=False)
    questions_df.to_csv(questions_filepath, index=False)

    # Create csv with each word as a row (word data)
    spacy_model_name = "nl_core_news_sm"
    words_filepath = "data/words.csv"
    syllabify, pos_tag, length, pvl = True, True, True, True
    # Try loading the model. If not found, download and load it
    try:
        nlp = spacy.load(spacy_model_name)
    except OSError:
        print(f"Model '{spacy_model_name}' not found. Downloading...")
        download(spacy_model_name)
        nlp = spacy.load(spacy_model_name)
    texts_df = pd.read_csv(texts_filepath)
    words_df = create_word_file(texts_df, nlp, syllabify, pos_tag, length, pvl)
    words_df = add_missing_syllables(words_df)
    words_df.to_csv(words_filepath)
    # check alignment between spacy words and open sesame words
    words_df = pd.read_csv(words_filepath)
    words_df_without_punct = processing_to_align_with_opensesame(words_df)
    words_df_without_punct.to_csv('data/words_without_punct.csv', index=False)
    os_df = pd.read_csv('data/word_coordinates_subject_0.csv')
    test_alignment_opensesame_words(words_df_without_punct, os_df)

    # Extract saliency values and added them as a column to word data
    language_model_name = "GroNLP/bert-base-dutch-cased" # "GroNLP/gpt2-small-dutch"
    saliency_path = f'data/{language_model_name.replace("/","_")}_saliency.csv'
    final_data_path = f'data/full_data_{language_model_name.replace("/","_")}.csv'
    texts_df = pd.read_csv('data/texts.csv')
    words_df = pd.read_csv('data/words.csv')
    saliency_df, words_plus_saliency_df = calculate_saliency_values(texts_df, words_df, language_model_name, saliency_path)
    saliency_df.to_csv(saliency_path, index=False)
    words_plus_saliency_df.to_csv(final_data_path, index=False)

if __name__ == "__main__":
    main()
