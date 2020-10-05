import requests
import pymongo
from tqdm import tqdm
import time
import datetime
from functools import reduce
import os
import markdown
import re

import mistune
import markdown
import re
import spacy
from spacy.lang.en.examples import sentences 

from nltk.stem import PorterStemmer
from string import punctuation
from tqdm import tqdm

punc = punctuation + u'.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：–'
md = markdown.Markdown()

nlp = spacy.load('en')
client = pymongo.MongoClient(host="192.168.0.106", port=29197)
client["github"].authenticate("github", "git332", "github")
db = client["github"]

db1 = db.all_step2_hastag
db2 = db.all_step3

def readme_preprocess(readme:str):
    
    # 解析 markdown
    html = markdown.markdown(readme)
    html = html.replace("\n","")
    html = re.sub(r'<code>.*?</code>',"",html)
    html = re.sub(r'```.*\n*.*\n*```', "", html)
    html = re.sub(r'```.*?\n*?.*?\n*?```', "", html)
    html = re.sub(r'```.*?```',"",html)
    # 去除html标签
    html = re.sub(r'(<.*?>)', "", html)
    html = html.lower()
    # 移除标点符号
    html = re.sub(r"[{}]+".format(punc), " ", html)
    doc = nlp(html)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
#     print(tokens[:3])
    readme_clean = ""
    
    st=PorterStemmer()
    for t in tokens:
        readme_clean = readme_clean + " "+ st.stem(t)
    # 去除emoji
    readme_clean = re.sub(
    r"\U0000231A|\U0000231B|\U000023E9|\U000023EA|\U000023EB|\U000023EC|\U000023F0|\U000023F3|\U000025FD|\U000025FE|\U00002614|\U00002615|\U00002648|\U00002649|\U0000264A|\U0000264B|\U0000264C|\U0000264D|\U0000264E|\U0000264F|\U00002650|\U00002651|\U00002652|\U00002653|\U0000267F|\U00002693|\U000026A1|\U000026AA|\U000026AB|\U000026BD|\U000026BE|\U000026C4|\U000026C5|\U000026CE|\U000026D4|\U000026EA|\U000026F2|\U000026F3|\U000026F5|\U000026FA|\U000026FD|\U00002705|\U0000270A|\U0000270B|\U00002728|\U0000274C|\U0000274E|\U00002753|\U00002754|\U00002755|\U00002757|\U00002795|\U00002796|\U00002797|\U000027B0|\U000027BF|\U00002B1B|\U00002B1C|\U00002B50|\U00002B55|\U0001F004|\U0001F0CF|\U0001F18E|\U0001F191|\U0001F192|\U0001F193|\U0001F194|\U0001F195|\U0001F196|\U0001F197|\U0001F198|\U0001F199|\U0001F19A|\U0001F201|\U0001F21A|\U0001F22F|\U0001F232|\U0001F233|\U0001F234|\U0001F235|\U0001F236|\U0001F238|\U0001F239|\U0001F23A|\U0001F250|\U0001F251|\U0001F300|\U0001F301|\U0001F302|\U0001F303|\U0001F304|\U0001F305|\U0001F306|\U0001F307|\U0001F308|\U0001F309|\U0001F30A|\U0001F30B|\U0001F30C|\U0001F30D|\U0001F30E|\U0001F30F|\U0001F310|\U0001F311|\U0001F312|\U0001F313|\U0001F314|\U0001F315|\U0001F316|\U0001F317|\U0001F318|\U0001F319|\U0001F31A|\U0001F31B|\U0001F31C|\U0001F31D|\U0001F31E|\U0001F31F|\U0001F320|\U0001F32D|\U0001F32E|\U0001F32F|\U0001F330|\U0001F331|\U0001F332|\U0001F333|\U0001F334|\U0001F335|\U0001F337|\U0001F338|\U0001F339|\U0001F33A|\U0001F33B|\U0001F33C|\U0001F33D|\U0001F33E|\U0001F33F|\U0001F340|\U0001F341|\U0001F342|\U0001F343|\U0001F344|\U0001F345|\U0001F346|\U0001F347|\U0001F348|\U0001F349|\U0001F34A|\U0001F34B|\U0001F34C|\U0001F34D|\U0001F34E|\U0001F34F|\U0001F350|\U0001F351|\U0001F352|\U0001F353|\U0001F354|\U0001F355|\U0001F356|\U0001F357|\U0001F358|\U0001F359|\U0001F35A|\U0001F35B|\U0001F35C|\U0001F35D|\U0001F35E|\U0001F35F|\U0001F360|\U0001F361|\U0001F362|\U0001F363|\U0001F364|\U0001F365|\U0001F366|\U0001F367|\U0001F368|\U0001F369|\U0001F36A|\U0001F36B|\U0001F36C|\U0001F36D|\U0001F36E|\U0001F36F|\U0001F370|\U0001F371|\U0001F372|\U0001F373|\U0001F374|\U0001F375|\U0001F376|\U0001F377|\U0001F378|\U0001F379|\U0001F37A|\U0001F37B|\U0001F37C|\U0001F37E|\U0001F37F|\U0001F380|\U0001F381|\U0001F382|\U0001F383|\U0001F384|\U0001F385|\U0001F386|\U0001F387|\U0001F388|\U0001F389|\U0001F38A|\U0001F38B|\U0001F38C|\U0001F38D|\U0001F38E|\U0001F38F|\U0001F390|\U0001F391|\U0001F392|\U0001F393|\U0001F3A0|\U0001F3A1|\U0001F3A2|\U0001F3A3|\U0001F3A4|\U0001F3A5|\U0001F3A6|\U0001F3A7|\U0001F3A8|\U0001F3A9|\U0001F3AA|\U0001F3AB|\U0001F3AC|\U0001F3AD|\U0001F3AE|\U0001F3AF|\U0001F3B0|\U0001F3B1|\U0001F3B2|\U0001F3B3|\U0001F3B4|\U0001F3B5|\U0001F3B6|\U0001F3B7|\U0001F3B8|\U0001F3B9|\U0001F3BA|\U0001F3BB|\U0001F3BC|\U0001F3BD|\U0001F3BE|\U0001F3BF|\U0001F3C0|\U0001F3C1|\U0001F3C2|\U0001F3C3|\U0001F3C4|\U0001F3C5|\U0001F3C6|\U0001F3C7|\U0001F3C8|\U0001F3C9|\U0001F3CA|\U0001F3CF|\U0001F3D0|\U0001F3D1|\U0001F3D2|\U0001F3D3|\U0001F3E0|\U0001F3E1|\U0001F3E2|\U0001F3E3|\U0001F3E4|\U0001F3E5|\U0001F3E6|\U0001F3E7|\U0001F3E8|\U0001F3E9|\U0001F3EA|\U0001F3EB|\U0001F3EC|\U0001F3ED|\U0001F3EE|\U0001F3EF|\U0001F3F0|\U0001F3F4|\U0001F3F8|\U0001F3F9|\U0001F3FA|\U0001F3FB|\U0001F3FC|\U0001F3FD|\U0001F3FE|\U0001F3FF|\U0001F400|\U0001F401|\U0001F402|\U0001F403|\U0001F404|\U0001F405|\U0001F406|\U0001F407|\U0001F408|\U0001F409|\U0001F40A|\U0001F40B|\U0001F40C|\U0001F40D|\U0001F40E|\U0001F40F|\U0001F410|\U0001F411|\U0001F412|\U0001F413|\U0001F414|\U0001F415|\U0001F416|\U0001F417|\U0001F418|\U0001F419|\U0001F41A|\U0001F41B|\U0001F41C|\U0001F41D|\U0001F41E|\U0001F41F|\U0001F420|\U0001F421|\U0001F422|\U0001F423|\U0001F424|\U0001F425|\U0001F426|\U0001F427|\U0001F428|\U0001F429|\U0001F42A|\U0001F42B|\U0001F42C|\U0001F42D|\U0001F42E|\U0001F42F|\U0001F430|\U0001F431|\U0001F432|\U0001F433|\U0001F434|\U0001F435|\U0001F436|\U0001F437|\U0001F438|\U0001F439|\U0001F43A|\U0001F43B|\U0001F43C|\U0001F43D|\U0001F43E|\U0001F440|\U0001F442|\U0001F443|\U0001F444|\U0001F445|\U0001F446|\U0001F447|\U0001F448|\U0001F449|\U0001F44A|\U0001F44B|\U0001F44C|\U0001F44D|\U0001F44E|\U0001F44F|\U0001F450|\U0001F451|\U0001F452|\U0001F453|\U0001F454|\U0001F455|\U0001F456|\U0001F457|\U0001F458|\U0001F459|\U0001F45A|\U0001F45B|\U0001F45C|\U0001F45D|\U0001F45E|\U0001F45F|\U0001F460|\U0001F461|\U0001F462|\U0001F463|\U0001F464|\U0001F465|\U0001F466|\U0001F467|\U0001F468|\U0001F469|\U0001F46A|\U0001F46B|\U0001F46C|\U0001F46D|\U0001F46E|\U0001F46F|\U0001F470|\U0001F471|\U0001F472|\U0001F473|\U0001F474|\U0001F475|\U0001F476|\U0001F477|\U0001F478|\U0001F479|\U0001F47A|\U0001F47B|\U0001F47C|\U0001F47D|\U0001F47E|\U0001F47F|\U0001F480|\U0001F481|\U0001F482|\U0001F483|\U0001F484|\U0001F485|\U0001F486|\U0001F487|\U0001F488|\U0001F489|\U0001F48A|\U0001F48B|\U0001F48C|\U0001F48D|\U0001F48E|\U0001F48F|\U0001F490|\U0001F491|\U0001F492|\U0001F493|\U0001F494|\U0001F495|\U0001F496|\U0001F497|\U0001F498|\U0001F499|\U0001F49A|\U0001F49B|\U0001F49C|\U0001F49D|\U0001F49E|\U0001F49F|\U0001F4A0|\U0001F4A1|\U0001F4A2|\U0001F4A3|\U0001F4A4|\U0001F4A5|\U0001F4A6|\U0001F4A7|\U0001F4A8|\U0001F4A9|\U0001F4AA|\U0001F4AB|\U0001F4AC|\U0001F4AD|\U0001F4AE|\U0001F4AF|\U0001F4B0|\U0001F4B1|\U0001F4B2|\U0001F4B3|\U0001F4B4|\U0001F4B5|\U0001F4B6|\U0001F4B7|\U0001F4B8|\U0001F4B9|\U0001F4BA|\U0001F4BB|\U0001F4BC|\U0001F4BD|\U0001F4BE|\U0001F4BF|\U0001F4C0|\U0001F4C1|\U0001F4C2|\U0001F4C3|\U0001F4C4|\U0001F4C5|\U0001F4C6|\U0001F4C7|\U0001F4C8|\U0001F4C9|\U0001F4CA|\U0001F4CB|\U0001F4CC|\U0001F4CD|\U0001F4CE|\U0001F4CF|\U0001F4D0|\U0001F4D1|\U0001F4D2|\U0001F4D3|\U0001F4D4|\U0001F4D5|\U0001F4D6|\U0001F4D7|\U0001F4D8|\U0001F4D9|\U0001F4DA|\U0001F4DB|\U0001F4DC|\U0001F4DD|\U0001F4DE|\U0001F4DF|\U0001F4E0|\U0001F4E1|\U0001F4E2|\U0001F4E3|\U0001F4E4|\U0001F4E5|\U0001F4E6|\U0001F4E7|\U0001F4E8|\U0001F4E9|\U0001F4EA|\U0001F4EB|\U0001F4EC|\U0001F4ED|\U0001F4EE|\U0001F4EF|\U0001F4F0|\U0001F4F1|\U0001F4F2|\U0001F4F3|\U0001F4F4|\U0001F4F5|\U0001F4F6|\U0001F4F7|\U0001F4F8|\U0001F4F9|\U0001F4FA|\U0001F4FB|\U0001F4FC|\U0001F4FF|\U0001F500|\U0001F501|\U0001F502|\U0001F503|\U0001F504|\U0001F505|\U0001F506|\U0001F507|\U0001F508|\U0001F509|\U0001F50A|\U0001F50B|\U0001F50C|\U0001F50D|\U0001F50E|\U0001F50F|\U0001F510|\U0001F511|\U0001F512|\U0001F513|\U0001F514|\U0001F515|\U0001F516|\U0001F517|\U0001F518|\U0001F519|\U0001F51A|\U0001F51B|\U0001F51C|\U0001F51D|\U0001F51E|\U0001F51F|\U0001F520|\U0001F521|\U0001F522|\U0001F523|\U0001F524|\U0001F525|\U0001F526|\U0001F527|\U0001F528|\U0001F529|\U0001F52A|\U0001F52B|\U0001F52C|\U0001F52D|\U0001F52E|\U0001F52F|\U0001F530|\U0001F531|\U0001F532|\U0001F533|\U0001F534|\U0001F535|\U0001F536|\U0001F537|\U0001F538|\U0001F539|\U0001F53A|\U0001F53B|\U0001F53C|\U0001F53D|\U0001F54B|\U0001F54C|\U0001F54D|\U0001F54E|\U0001F550|\U0001F551|\U0001F552|\U0001F553|\U0001F554|\U0001F555|\U0001F556|\U0001F557|\U0001F558|\U0001F559|\U0001F55A|\U0001F55B|\U0001F55C|\U0001F55D|\U0001F55E|\U0001F55F|\U0001F560|\U0001F561|\U0001F562|\U0001F563|\U0001F564|\U0001F565|\U0001F566|\U0001F567|\U0001F595|\U0001F596|\U0001F5FB|\U0001F5FC|\U0001F5FD|\U0001F5FE|\U0001F5FF|\U0001F600|\U0001F601|\U0001F602|\U0001F603|\U0001F604|\U0001F605|\U0001F606|\U0001F607|\U0001F608|\U0001F609|\U0001F60A|\U0001F60B|\U0001F60C|\U0001F60D|\U0001F60E|\U0001F60F|\U0001F610|\U0001F611|\U0001F612|\U0001F613|\U0001F614|\U0001F615|\U0001F616|\U0001F617|\U0001F618|\U0001F619|\U0001F61A|\U0001F61B|\U0001F61C|\U0001F61D|\U0001F61E|\U0001F61F|\U0001F620|\U0001F621|\U0001F622|\U0001F623|\U0001F624|\U0001F625|\U0001F626|\U0001F627|\U0001F628|\U0001F629|\U0001F62A|\U0001F62B|\U0001F62C|\U0001F62D|\U0001F62E|\U0001F62F|\U0001F630|\U0001F631|\U0001F632|\U0001F633|\U0001F634|\U0001F635|\U0001F636|\U0001F637|\U0001F638|\U0001F639|\U0001F63A|\U0001F63B|\U0001F63C|\U0001F63D|\U0001F63E|\U0001F63F|\U0001F640|\U0001F641|\U0001F642|\U0001F643|\U0001F644|\U0001F645|\U0001F646|\U0001F647|\U0001F648|\U0001F649|\U0001F64A|\U0001F64B|\U0001F64C|\U0001F64D|\U0001F64E|\U0001F64F|\U0001F680|\U0001F681|\U0001F682|\U0001F683|\U0001F684|\U0001F685|\U0001F686|\U0001F687|\U0001F688|\U0001F689|\U0001F68A|\U0001F68B|\U0001F68C|\U0001F68D|\U0001F68E|\U0001F68F|\U0001F690|\U0001F691|\U0001F692|\U0001F693|\U0001F694|\U0001F695|\U0001F696|\U0001F697|\U0001F698|\U0001F699|\U0001F69A|\U0001F69B|\U0001F69C|\U0001F69D|\U0001F69E|\U0001F69F|\U0001F6A0|\U0001F6A1|\U0001F6A2|\U0001F6A3|\U0001F6A4|\U0001F6A5|\U0001F6A6|\U0001F6A7|\U0001F6A8|\U0001F6A9|\U0001F6AA|\U0001F6AB|\U0001F6AC|\U0001F6AD|\U0001F6AE|\U0001F6AF|\U0001F6B0|\U0001F6B1|\U0001F6B2|\U0001F6B3|\U0001F6B4|\U0001F6B5|\U0001F6B6|\U0001F6B7|\U0001F6B8|\U0001F6B9|\U0001F6BA|\U0001F6BB|\U0001F6BC|\U0001F6BD|\U0001F6BE|\U0001F6BF|\U0001F6C0|\U0001F6C1|\U0001F6C2|\U0001F6C3|\U0001F6C4|\U0001F6C5|\U0001F6CC|\U0001F6D0|\U0001F6EB|\U0001F6EC|\U0001F910|\U0001F911|\U0001F912|\U0001F913|\U0001F914|\U0001F915|\U0001F916|\U0001F917|\U0001F918|\U0001F980|\U0001F981|\U0001F982|\U0001F983|\U0001F984|\U0001F9C0|\U0001F1E6\U0001F1E8|\U0001F1E6\U0001F1E9|\U0001F1E6\U0001F1EA|\U0001F1E6\U0001F1EB|\U0001F1E6\U0001F1EC|\U0001F1E6\U0001F1EE|\U0001F1E6\U0001F1F1|\U0001F1E6\U0001F1F2|\U0001F1E6\U0001F1F4|\U0001F1E6\U0001F1F6|\U0001F1E6\U0001F1F7|\U0001F1E6\U0001F1F8|\U0001F1E6\U0001F1F9|\U0001F1E6\U0001F1FA|\U0001F1E6\U0001F1FC|\U0001F1E6\U0001F1FD|\U0001F1E6\U0001F1FF|\U0001F1E7\U0001F1E6|\U0001F1E7\U0001F1E7|\U0001F1E7\U0001F1E9|\U0001F1E7\U0001F1EA|\U0001F1E7\U0001F1EB|\U0001F1E7\U0001F1EC|\U0001F1E7\U0001F1ED|\U0001F1E7\U0001F1EE|\U0001F1E7\U0001F1EF|\U0001F1E7\U0001F1F1|\U0001F1E7\U0001F1F2|\U0001F1E7\U0001F1F3|\U0001F1E7\U0001F1F4|\U0001F1E7\U0001F1F6|\U0001F1E7\U0001F1F7|\U0001F1E7\U0001F1F8|\U0001F1E7\U0001F1F9|\U0001F1E7\U0001F1FB|\U0001F1E7\U0001F1FC|\U0001F1E7\U0001F1FE|\U0001F1E7\U0001F1FF|\U0001F1E8\U0001F1E6|\U0001F1E8\U0001F1E8|\U0001F1E8\U0001F1E9|\U0001F1E8\U0001F1EB|\U0001F1E8\U0001F1EC|\U0001F1E8\U0001F1ED|\U0001F1E8\U0001F1EE|\U0001F1E8\U0001F1F0|\U0001F1E8\U0001F1F1|\U0001F1E8\U0001F1F2|\U0001F1E8\U0001F1F3|\U0001F1E8\U0001F1F4|\U0001F1E8\U0001F1F5|\U0001F1E8\U0001F1F7|\U0001F1E8\U0001F1FA|\U0001F1E8\U0001F1FB|\U0001F1E8\U0001F1FC|\U0001F1E8\U0001F1FD|\U0001F1E8\U0001F1FE|\U0001F1E8\U0001F1FF|\U0001F1E9\U0001F1EA|\U0001F1E9\U0001F1EC|\U0001F1E9\U0001F1EF|\U0001F1E9\U0001F1F0|\U0001F1E9\U0001F1F2|\U0001F1E9\U0001F1F4|\U0001F1E9\U0001F1FF|\U0001F1EA\U0001F1E6|\U0001F1EA\U0001F1E8|\U0001F1EA\U0001F1EA|\U0001F1EA\U0001F1EC|\U0001F1EA\U0001F1ED|\U0001F1EA\U0001F1F7|\U0001F1EA\U0001F1F8|\U0001F1EA\U0001F1F9|\U0001F1EA\U0001F1FA|\U0001F1EB\U0001F1EE|\U0001F1EB\U0001F1EF|\U0001F1EB\U0001F1F0|\U0001F1EB\U0001F1F2|\U0001F1EB\U0001F1F4|\U0001F1EB\U0001F1F7|\U0001F1EC\U0001F1E6|\U0001F1EC\U0001F1E7|\U0001F1EC\U0001F1E9|\U0001F1EC\U0001F1EA|\U0001F1EC\U0001F1EB|\U0001F1EC\U0001F1EC|\U0001F1EC\U0001F1ED|\U0001F1EC\U0001F1EE|\U0001F1EC\U0001F1F1|\U0001F1EC\U0001F1F2|\U0001F1EC\U0001F1F3|\U0001F1EC\U0001F1F5|\U0001F1EC\U0001F1F6|\U0001F1EC\U0001F1F7|\U0001F1EC\U0001F1F8|\U0001F1EC\U0001F1F9|\U0001F1EC\U0001F1FA|\U0001F1EC\U0001F1FC|\U0001F1EC\U0001F1FE|\U0001F1ED\U0001F1F0|\U0001F1ED\U0001F1F2|\U0001F1ED\U0001F1F3|\U0001F1ED\U0001F1F7|\U0001F1ED\U0001F1F9|\U0001F1ED\U0001F1FA|\U0001F1EE\U0001F1E8|\U0001F1EE\U0001F1E9|\U0001F1EE\U0001F1EA|\U0001F1EE\U0001F1F1|\U0001F1EE\U0001F1F2|\U0001F1EE\U0001F1F3|\U0001F1EE\U0001F1F4|\U0001F1EE\U0001F1F6|\U0001F1EE\U0001F1F7|\U0001F1EE\U0001F1F8|\U0001F1EE\U0001F1F9|\U0001F1EF\U0001F1EA|\U0001F1EF\U0001F1F2|\U0001F1EF\U0001F1F4|\U0001F1EF\U0001F1F5|\U0001F1F0\U0001F1EA|\U0001F1F0\U0001F1EC|\U0001F1F0\U0001F1ED|\U0001F1F0\U0001F1EE|\U0001F1F0\U0001F1F2|\U0001F1F0\U0001F1F3|\U0001F1F0\U0001F1F5|\U0001F1F0\U0001F1F7|\U0001F1F0\U0001F1FC|\U0001F1F0\U0001F1FE|\U0001F1F0\U0001F1FF|\U0001F1F1\U0001F1E6|\U0001F1F1\U0001F1E7|\U0001F1F1\U0001F1E8|\U0001F1F1\U0001F1EE|\U0001F1F1\U0001F1F0|\U0001F1F1\U0001F1F7|\U0001F1F1\U0001F1F8|\U0001F1F1\U0001F1F9|\U0001F1F1\U0001F1FA|\U0001F1F1\U0001F1FB|\U0001F1F1\U0001F1FE|\U0001F1F2\U0001F1E6|\U0001F1F2\U0001F1E8|\U0001F1F2\U0001F1E9|\U0001F1F2\U0001F1EA|\U0001F1F2\U0001F1EB|\U0001F1F2\U0001F1EC|\U0001F1F2\U0001F1ED|\U0001F1F2\U0001F1F0|\U0001F1F2\U0001F1F1|\U0001F1F2\U0001F1F2|\U0001F1F2\U0001F1F3|\U0001F1F2\U0001F1F4|\U0001F1F2\U0001F1F5|\U0001F1F2\U0001F1F6|\U0001F1F2\U0001F1F7|\U0001F1F2\U0001F1F8|\U0001F1F2\U0001F1F9|\U0001F1F2\U0001F1FA|\U0001F1F2\U0001F1FB|\U0001F1F2\U0001F1FC|\U0001F1F2\U0001F1FD|\U0001F1F2\U0001F1FE|\U0001F1F2\U0001F1FF|\U0001F1F3\U0001F1E6|\U0001F1F3\U0001F1E8|\U0001F1F3\U0001F1EA|\U0001F1F3\U0001F1EB|\U0001F1F3\U0001F1EC|\U0001F1F3\U0001F1EE|\U0001F1F3\U0001F1F1|\U0001F1F3\U0001F1F4|\U0001F1F3\U0001F1F5|\U0001F1F3\U0001F1F7|\U0001F1F3\U0001F1FA|\U0001F1F3\U0001F1FF|\U0001F1F4\U0001F1F2|\U0001F1F5\U0001F1E6|\U0001F1F5\U0001F1EA|\U0001F1F5\U0001F1EB|\U0001F1F5\U0001F1EC|\U0001F1F5\U0001F1ED|\U0001F1F5\U0001F1F0|\U0001F1F5\U0001F1F1|\U0001F1F5\U0001F1F2|\U0001F1F5\U0001F1F3|\U0001F1F5\U0001F1F7|\U0001F1F5\U0001F1F8|\U0001F1F5\U0001F1F9|\U0001F1F5\U0001F1FC|\U0001F1F5\U0001F1FE|\U0001F1F6\U0001F1E6|\U0001F1F7\U0001F1EA|\U0001F1F7\U0001F1F4|\U0001F1F7\U0001F1F8|\U0001F1F7\U0001F1FA|\U0001F1F7\U0001F1FC|\U0001F1F8\U0001F1E6|\U0001F1F8\U0001F1E7|\U0001F1F8\U0001F1E8|\U0001F1F8\U0001F1E9|\U0001F1F8\U0001F1EA|\U0001F1F8\U0001F1EC|\U0001F1F8\U0001F1ED|\U0001F1F8\U0001F1EE|\U0001F1F8\U0001F1EF|\U0001F1F8\U0001F1F0|\U0001F1F8\U0001F1F1|\U0001F1F8\U0001F1F2|\U0001F1F8\U0001F1F3|\U0001F1F8\U0001F1F4|\U0001F1F8\U0001F1F7|\U0001F1F8\U0001F1F8|\U0001F1F8\U0001F1F9|\U0001F1F8\U0001F1FB|\U0001F1F8\U0001F1FD|\U0001F1F8\U0001F1FE|\U0001F1F8\U0001F1FF|\U0001F1F9\U0001F1E6|\U0001F1F9\U0001F1E8|\U0001F1F9\U0001F1E9|\U0001F1F9\U0001F1EB|\U0001F1F9\U0001F1EC|\U0001F1F9\U0001F1ED|\U0001F1F9\U0001F1EF|\U0001F1F9\U0001F1F0|\U0001F1F9\U0001F1F1|\U0001F1F9\U0001F1F2|\U0001F1F9\U0001F1F3|\U0001F1F9\U0001F1F4|\U0001F1F9\U0001F1F7|\U0001F1F9\U0001F1F9|\U0001F1F9\U0001F1FB|\U0001F1F9\U0001F1FC|\U0001F1F9\U0001F1FF|\U0001F1FA\U0001F1E6|\U0001F1FA\U0001F1EC|\U0001F1FA\U0001F1F2|\U0001F1FA\U0001F1F8|\U0001F1FA\U0001F1FE|\U0001F1FA\U0001F1FF|\U0001F1FB\U0001F1E6|\U0001F1FB\U0001F1E8|\U0001F1FB\U0001F1EA|\U0001F1FB\U0001F1EC|\U0001F1FB\U0001F1EE|\U0001F1FB\U0001F1F3|\U0001F1FB\U0001F1FA|\U0001F1FC\U0001F1EB|\U0001F1FC\U0001F1F8|\U0001F1FD\U0001F1F0|\U0001F1FE\U0001F1EA|\U0001F1FE\U0001F1F9|\U0001F1FF\U0001F1E6|\U0001F1FF\U0001F1F2|\U0001F1FF\U0001F1FC",
    " ", readme_clean)
    # 去除多余空格
    readme_clean = re.sub(" +", " ", readme_clean)
    return readme_clean


i = 0
for d in tqdm(list(db1.find())[12500:]):
    i+=1
    path = os.path.join(os.path.join("..", "git_data"),
                        reduce(os.path.join, d['link'].split("/")))
    if "readme_file" in d.keys() and d["readme_file"] == 1:
        with open(os.path.join(path, "readme.md"), 'r', encoding='utf-8') as f:
            d["readme_parser"] = readme_preprocess(f.read()).strip()
    else:
        d["readme_parser"] = ""
    
    del d["_id"]
    db2.insert(d)

print("done {}".format(i))