
import xml.etree.ElementTree as et

file = open("train_data/tourkikes_agglikes.txt")

critiques = file.readlines()
lista = []
for item in critiques:
    if item != "\n":
        lista.append(item.strip())

tree = et.parse("train_data/train_english.xml")
root = tree.getroot()
critiques = []
critique_to_opinion = []
test = False
for review in root:
    for sentences in review:
        for sentence in sentences:
            opinions = sentence.find("Opinions")
            if test:
                sentence.find("text").text = lista[0]
                lista = lista[1:]
            if opinions:
                if not test:
                    critiques.append(sentence.find("text").text)
                    sentence.find("text").text = lista[0]
                    lista = lista[1:]
                critique_to_opinion.append([])
                for opinion in opinions:
                    tag = opinion.attrib
                    critique_to_opinion[-1].append(tag)

tree.write("train_data/train_turkish.xml")