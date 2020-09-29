import os
import re
import numpy as np
import json
import pagexml
import pandas as pd

from metadata import extract_last_dir, load_files

def create_string(text_list:list):
    text = ""
    for item in text_list:
        text += str(item)
    return text

def get_text_from_xml(pxml):
    text_values = []
    for tl in pxml.select(".//_:TextLine"):
        for word in pxml.select(".//_:Word", tl):
            text_values.append(pxml.getTextEquiv(word))
        text = create_string(text_values)       
    return text


path = "~/master_thesis/data/SCA001"
temp = load_files(path)
paths = []
for path in temp:
    if("validated" in path.split('/')[-1]):
        paths.append(path)

df = pd.DataFrame(columns=["uuid", "filename", "sentence", "label"])

for path in paths:
    pxml = pagexml.PageXML(path)
    document = pxml.selectNth(".//_:PcGts", 0)
    print(document)
    uuid = pxml.getPropertyValue(document, "uuid")
    filename = pxml.getPropertyValue(document, "fileName")
    label = pxml.getPropertyValue(document, "class")
    ocr_path = "~/master_thesis/data/SCA001" + '/' + str(uuid) + '/' + 'ocr_page.xml'
    pxml_ocr = pagexml.PageXML(path_ocr)
    sentence = get_text_from_xml(pxml_ocr)
    df = df.append({"uuid": str(uuid), "filename": filename, "sentence": sentence, "label": label}, ignore_index=True)