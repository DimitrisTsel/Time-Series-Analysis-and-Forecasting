# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 15:21:16 2021

@author: DimTsel
"""
'''This is a parser which constructed to parse a large xml dblp
file and to get the number of publications by year of the field
of Computational Linguistics'''


file = r'C:\Users\30694\Desktop\Dblp\dblp-2021-02-01.xml'
final=r'C:\Users\30694\Desktop\Dblp\dblp_data.txt'
with open(file, 'r') as f:
    count = dict()
    total = 0
    #keywords related to computational linguistics
    words = ["linguistics", "linguistic","computational linguistic","computational linguistics","speech processing","machine translation","natural language process","nlp","computer based translation","translating machine","lexical resources","speech recognition","translation memory","computational lexicology","speech synthesis","social media mining","text editor","grammar correction","word processing","spelling correction"]
    for line in f:
        if '<title>' in line:
            for word in words:
                #we are searching the words in lower case
                if word.lower() in line.lower():
                    total+=1
                    print(total,"\t", line[7:-9])
                    #we are looking for the years, up to four lines under the titles and we add a publication to its year each time
                    first=f.readline()
                    second=f.readline()
                    third=f.readline()
                    fourth=f.readline()
                    if '<year>' in first:
                        if first[6:10] not in count:
                            count[first[6:10]]=1
                        else:
                            count[first[6:10]]+=1
                    if '<year>' in second:
                        if second[6:10] not in count:
                            count[second[6:10]]=1
                        else:
                            count[second[6:10]]+=1
                    if '<year>' in third:
                        if third[6:10] not in count:
                            count[third[6:10]]=1
                        else:
                            count[third[6:10]]+=1
                    if '<year>' in fourth:
                        if fourth[6:10] not in count:
                            count[fourth[6:10]]=1
                        else:
                            count[fourth[6:10]]+=1
                    break
total_years=0
#sorting data by year
for i in sorted(count):
    print(i,'\t', count[i], end='\n')
    total_years+=count[i]
print(total_years)

'''
#store data to txt file
f = open(final, "w")
f.write("Years" + "\t" + "Number of Publications\n")
for i in sorted(count):
   f.write((str(i) + "\t" + str(count[i]) + "\n"))
f.close()
'''
