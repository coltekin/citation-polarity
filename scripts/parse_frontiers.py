#!/usr/bin/env python3
"""Parse frontiers XML data, convert it to simple TSV file(s).
"""

import sys
import regex
import csv
import os
from multiprocessing import Pool
from lxml import etree
from sentence_splitter import SentenceSplitter

author_re = [
    regex.compile(r"^.*\b(?P<a>[[:upper:]][\w'’-]+(\s[[:upper:]]\.)?,?\set\sal\.('s)?[,.]?(\s\()?[\s(]+)$"),
    regex.compile(r"^.*\b(?P<a>[[:upper:]][\w'’-]+\s+and\s+[[:upper:]][\w'’-]+[.]?,?(\s\()?[\s(]+)$"),
    regex.compile(r"^.*[^.?!]\s(?P<a>[[:upper:]][\w’'-]+(\s[[:upper:]]\.)?\s[[:upper:]][\w'’-]+(\s[[:upper:]]\.)?\s[[:upper:]][\w'’-]+(\s[[:upper:]]\.)?[.]?,?(\s\()?[\s(]+)$"),
    regex.compile(r"^.*[^.?!]\s(?P<a>[[:upper:]][\w’'-]+(\s[[:upper:]]\.)?\s(of\s)?[[:upper:]][\w'’-]+(\s[[:upper:]]\.)?[.]?,?(\s\()?[\s(]+)$"),
    regex.compile(r"^.*\b(?P<a>[[:upper:]][\w'’-]+(\s[[:upper:]]\.)?,?(\s\()?[\s(]+)$"),
    regex.compile(r"^.*\((?P<a>([\w'’-]+\s){1,3}[\w'’-]+\[.,]?(\s\()?(\(\w{2-6}\))?[\s(]+)$"),
]

year_re = regex.compile(
    r'(\(?(\d{4}(\w(,\w(,\w)?)?)?(\s?\[\d{4}(\s?–\s?\d{4})?\]|\s?[/-]\s?\d{2,4})?(,?\s*p\.\s*\d+)?)\)?'
      r'|\(?([fF]orthcoming|[aA]ccepted|[Ii]n\spress|n\.d(\.)?\)?)'
    r')$')

multicite_re = regex.compile(r'\d{4}(\w(,\w(,\w)?)?)?[,;]\s?$')

#        |())'
#        r',(\s\()?\s+)$')

#author_re = regex.compile(
#        r'^.*\b(?P<a>|([[:upper:]]\w+ et al.)'
#        r',(\s\()?\s+)$')

def get_fulltext(el):
    citations = []
    text = ""
    if el.text:
        text = el.text.replace('\n', '').replace('\r', '')
    for child in el.getchildren():
        if child.tag == 'xref' and child.get('ref-type') == "bibr":
            if not child.text: continue
            if year_re.match(child.text): # reference only includes year
                year = child.text
                authors = None
                for i, a_re in enumerate(author_re):
                    m = a_re.match(text)
                    if m:
                        authors = m.group('a')
                        break
#                print('-----', i, authors, '---------', text)
                if not authors and multicite_re.search(text) and len(citations) > 0:
#                    print(f"Trying to recover: _{text[-30:]}_")
                    authors = citations[-1][0]
                if authors and year:
                    citations.append((authors.strip(' ('), year.strip(' ()'), len(text)))
#                    print(f"X: _{authors}_ _{year}_")
                else:
                    print(f"Cannot match: _{text[-30:]}_ _{year}_")
            elif child.text.strip() in {'b', 'c', 'd'}:
                if len(citations) > 0 and citations[-1][1] in {'a', 'b', 'c'}:
                    authors = citations[-1][0]
                    year = citations[-1][:-1]  + child.text.strip()
                    citations.append((authors.strip(' ('), year.strip(' ()'), len(text)))
            elif year_re.search(child.text): # assume that the whole reference string is the citation
                authors = year_re.sub('', child.text).strip()
                if not authors and len(citations) > 0:
                    authors = citations[-1][0]
                year = year_re.search(child.text).group(1)
                citations.append((authors.strip(' ('), year.strip(' ()'), len(text)))
#                print(f"Y: _{child.text}_ _{authors}_ _{year}_")
            else:
                print(f"Failed to match: _{text[-30:]}_ _{child.text}_")
        elif child.tag in {'disp-formula', 'table-wrap'}:
            continue
        ctext, ccit = get_fulltext(child)
        citations.extend(ccit)
        tailt = ""
        if child.tail:
            tailt = child.tail.replace('\n', '').replace('\r', '')
        text += ctext + tailt
    return text, citations

def parse_frontiers(filename):
    try:
        with open(filename, 'rb') as f:
            t = etree.parse(f)
    except:
        return [], [], None
    splitter = SentenceSplitter(language='en')
    splitter._SentenceSplitter__non_breaking_prefixes['p'] = SentenceSplitter.PrefixType.DEFAULT
    splitter._SentenceSplitter__non_breaking_prefixes['pp'] = SentenceSplitter.PrefixType.DEFAULT
    root = t.getroot()
    date = root.find('.//front/article-meta/pub-date')
    year = date.find('./year').text
    month = date.find('./month').text
    day = date.find('./day').text
    b = root.find('./body')
    pubdate = '-'.join((year, month, day))
    article, citations = [], []
    for p in b.xpath('.//p'):
        if p.getparent().tag != 'sec':
#            print('Skipping non-section paragraph.')
            continue
        txt, cit = get_fulltext(p)
        article.append(splitter.split(txt))
        citations.append(cit)

    return article, citations, pubdate

def process_file(fname):
    print(fname)
    aid = os.path.basename(fname).replace('.xml', '')
    d = []
    article, citations, pubdate = parse_frontiers(fname)
    for i, parcit in enumerate(citations):
        if len(parcit) == 0: continue
        citn, poffset = 0, 0
        citoffset = parcit[citn][2]
        for j, sent in enumerate(article[i]):
            if citn >= len(parcit): break
            while poffset + len(sent) > citoffset:
                cit = ' '.join(parcit[citn][:2])
                context = sent
                if j > 0:
                    context = " ".join((article[i][j-1], context))
                if j < len(article[i]) - 1:
                    context = " ".join((context, article[i][j+1]))
                if parcit[citn][0] in context:
                    d.append((aid, pubdate, cit, sent, context))
                else:
                    print(f"{cit} not found")
                citn += 1
                if citn >= len(parcit): break
                citoffset = parcit[citn][2]
            poffset += len(sent)
    return d


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('input', nargs="+")
    ap.add_argument('--output', '-O')
    args = ap.parse_args()

    if args.output is None:
        args.output = "output.tsv"
        head, tail = os.path.split(args.input[0])
        if head:
            args.output = os.path.basename(head) + '.tsv'

    p = Pool(processes=40)
    result = p.map(process_file, args.input)

    with open(args.output, 'wt') as fout:
        csvw = csv.writer(fout, delimiter='\t')
        csvw.writerow(('article', 'pubdate', 'citation', 'sentence', 'context'))
        for d in result:
            for row in d:
                csvw.writerow(row)
