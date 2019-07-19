#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 22:15:21 2019

@author: mohitbeniwal
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import accuracy_score 
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from numpy.random import uniform 
from pandas import ExcelWriter

#file_dir = os.path.join('/','Users', 'mohitbeniwal','Downloads','Sem 2','ML') 
#file = os.path.join(file_dir, 'rockpaperseaser.xlsx')
xl=pd.ExcelFile("rockpaperseaser.xlsx")
df=xl.parse("Sheet1")  
#df_win=df.loc[df['Result'] == 'win']
#df_loss=df.loc[df['Result'] == 'loss']
#df_tie=df.loc[df['Result'] == 'tie']
#df_r=df.loc[df['my pick'] == 'r']
#df_s=df.loc[df['my pick'] == 's']
#df_p=df.loc[df['my pick'] == 'p']

df=df.replace('r',1)
df=df.replace('p',2)
df=df.replace('s',3)




#df = pd.read_xlsx(file_name)
l_d=[]
for i in range(4,len(df)):
    l_d.append([df['my pick'][i],df['my pick'][i-1],df['my pick'][i-2],df['my pick'][i-3],df['my pick'][i-4],df['computer pick'][i],df['computer pick'][i-1],df['computer pick'][i-2],df['computer pick'][i-3],df['computer pick'][i-4]])
df1=pd.DataFrame(l_d,columns=['my_pick','my_p_1','my_p_2','my_p_3','my_p_4','computer_pick','com_p_1','com_p_2','com_p_3','com_p_4'])

df1['p_pred'] = ( df1['computer_pick'] == 2 )*1

df1['r_pred'] = ( df1['computer_pick'] == 1 )*1

df1['s_pred'] = ( df1['computer_pick'] == 3 )*1

#Decisoin tree for binaries (r_pred, s_pres, p_pred)

#p_pred



df_X = df1.drop({'p_pred','r_pred','s_pred','my_pick','computer_pick','my_p_4','com_p_4','my_p_3','com_p_3'}, axis = 1)

enc = OneHotEncoder()

# 2. FIT
enc.fit(df_X)

# 3. Transform
onehotlabels = enc.transform(df_X).toarray()
onehotlabels.shape

rf_p = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf_r = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf_s = RandomForestRegressor(n_estimators = 1000, random_state = 42)



X_train, X_test, y_train, y_test = train_test_split(onehotlabels, df1.p_pred, test_size = 0.4, random_state=1)
rf_p.fit(X_train, y_train);
p_pred = rf_p.predict(X_test)
accuracy_score(y_test, p_pred > 0.5 )

X_train, X_test, y_train, y_test = train_test_split(onehotlabels, df1.r_pred, test_size = 0.4, random_state=1)
rf_r.fit(X_train, y_train);
r_pred = rf_r.predict(X_test)
accuracy_score(y_test, r_pred > 0.5 )

X_train, X_test, y_train, y_test = train_test_split(onehotlabels, df1.s_pred, test_size = 0.4, random_state=1)
rf_s.fit(X_train, y_train);
s_pred = rf_s.predict(X_test)
accuracy_score(y_test, s_pred > 0.5 )

X_train, X_test, y_train, y_test = train_test_split(onehotlabels, df1.computer_pick, test_size = 0.4, random_state=1)

df_prs=pd.DataFrame()
df_prs['r_pred']=r_pred
df_prs['p_pred']=p_pred
df_prs['s_pred']=s_pred
df_prs['y_test']=y_test.reset_index().computer_pick
l_max=[]

for i in range(0,len(df_prs)):
    if(df_prs['r_pred'][i]>df_prs['p_pred'][i] and df_prs['r_pred'][i]>df_prs['s_pred'][i]):
        l_max.append(1)
    elif(df_prs['p_pred'][i]>df_prs['r_pred'][i] and df_prs['p_pred'][i]>df_prs['s_pred'][i]):
        l_max.append(2)
    elif(df_prs['s_pred'][i]>df_prs['r_pred'][i] and df_prs['s_pred'][i]>df_prs['p_pred'][i]):
        l_max.append(3)
df_prs['max']=l_max

sum(df_prs['y_test'] == df_prs['max'])


#Prediction on new game

#file_dir = os.path.join('/','Users', 'mohitbeniwal','Downloads','Sem 2','ML') 
#file = os.path.join(file_dir, 'rockpaperseaser_new.xlsx')
def main_f():
    xl1=pd.ExcelFile("rockpaperseaser_new1.xlsx")
    df=xl1.parse("Sheet1")  
    
    df=df.replace('r',1)
    df=df.replace('p',2)
    df=df.replace('s',3)
    
    df[[]]
    
    out_priority1=''
    if(df['Result'][len(df)-1]=='loss' and df['Result'][len(df)-2]=='loss'):
        i=random.randint(1,3)
        if(i==1):
            out_priority1='r'
        elif(i==2):
            out_priority1='p'
        else:
            out_priority1='s'
    
    out_priority2=''    
    if (df['my pick'][len(df)-1] == df['my pick'][len(df)-2]):
            if(df['my pick'][len(df)-1] == 1):
                out_priority2='s'
            elif (df['my pick'][len(df)-1] == 2):
                out_priority2='r'
            else :
                out_priority2='p'
    
    #df = pd.read_xlsx(file_name)
    l_d=[]
    for i in range(4,len(df)):
        l_d.append([df['my pick'][i],df['my pick'][i-1],df['my pick'][i-2],df['my pick'][i-3],df['my pick'][i-4],df['computer pick'][i],df['computer pick'][i-1],df['computer pick'][i-2],df['computer pick'][i-3],df['computer pick'][i-4]])
    df1=pd.DataFrame(l_d,columns=['my_pick','my_p_1','my_p_2','my_p_3','my_p_4','computer_pick','com_p_1','com_p_2','com_p_3','com_p_4'])
    
    df1['p_pred'] = ( df1['computer_pick'] == 2 )*1
    
    df1['r_pred'] = ( df1['computer_pick'] == 1 )*1
    
    df1['s_pred'] = ( df1['computer_pick'] == 3 )*1
    
    
    
    
    
    df_X = df1.drop({'p_pred','r_pred','s_pred','my_pick','computer_pick','my_p_4','com_p_4','my_p_3','com_p_3'}, axis = 1)
    
    # 2. FIT
    
    # 3. Transform
    onehotlabels = enc.transform(df_X).toarray()
    new_data = onehotlabels[onehotlabels.shape[0]-1]
    
    p_pred = rf_p.predict(new_data.reshape(1,-1))
    r_pred = rf_r.predict(new_data.reshape(1,-1))
    s_pred = rf_s.predict(new_data.reshape(1,-1))
    
    
    #adding randomness in draw
    cond_prob = p_pred + r_pred + s_pred ; 
    
    
    
    x = uniform(low = 0, high = 1, size = 1)
    
    sel = [p_pred/cond_prob ,  (p_pred +r_pred)/cond_prob , (p_pred + r_pred + s_pred)/cond_prob ]
    out_priority3=''
    if(x < sel[0]):
        out_priority3='s'
    elif(x < sel[1]):
        out_priority3='p'
    else :
        out_priority3='r'
    my=''
    if(out_priority1!=''):
        print('play with '+out_priority1)
        my=out_priority1
    elif(out_priority2!=''):
        print('play with '+out_priority2)
        my=out_priority2
    else:
        print('play with '+out_priority3)
        my=out_priority3
    #my = input('What did you play?(r,p,s) ')
    computer = input('What did the AI play?(r,p,s) ')
    res= input('Did you win(y/n/t) ')
    result=''
        
    if(res=='y'):
        result='win'
    elif(res=='n'):
        result='loss'
    else:
        result='tie'
        
    usr=[[my,computer,result]]
    usr_input=pd.DataFrame(usr,columns=['my pick','computer pick','Result'])
    final=xl1.parse("Sheet1")
    final=final.append(usr_input)
    #final.reset_index(drop=True)
    
    writer = ExcelWriter('rockpaperseaser_new1.xlsx')
    final.to_excel(writer,'Sheet1',index=False)
    writer.save()

on_off=''
while(on_off=='' or on_off=='y'):
    main_f()
    on_off = input('Want to play more?(y,n) ')
    



