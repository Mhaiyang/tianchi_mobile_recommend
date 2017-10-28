
# coding: utf-8

# In[1]:


import pandas as pd
import datetime
import matplotlib
from matplotlib import pyplot as plt
import seaborn
import numpy as np
get_ipython().magic('matplotlib inline')
matplotlib.rcParams['figure.figsize'] = (16, 5)


# 1.读取原始数据

# In[2]:


user_df=pd.read_csv('../data/tianchi_mobile_recommend_train_user.csv')
item_df=pd.read_csv('../data/tianchi_mobile_recommend_train_item.csv')


# In[3]:


user_df.head()


# In[4]:


item_df.head()


# 2.查看数据总量

# In[5]:


print('用户数量：',user_df.user_id.unique().shape[0])
print('商品总数量：',user_df.item_id.unique().shape[0])
print('操作记录总数：',len(user_df))
print('要预测的商品总数量:',item_df.item_id.unique().shape[0])


# In[6]:


# 只预测这个set中出现的商品
item_id_set=set(item_df.item_id)
print('要预测的商品总数量:',len(item_id_set))


# 3.数据格式转换(时间转换)

# In[7]:


get_ipython().run_cell_magic('time', '', 'user_df[\'day\']=user_df.time.apply(lambda x:datetime.datetime.strptime(x[:-3], "%Y-%m-%d"))\nuser_df[\'hour\']=user_df.time.apply(lambda x:int(x[-2:]))')


# 4.做一些数据统计

# In[8]:


print('每种行为的数量')
pd.DataFrame(user_df.behavior_type.value_counts())


# In[9]:


print('每天的行为数量')
pd.DataFrame(user_df.day.value_counts()).plot()


# 5.将12月17日加入购物车的商品作为预测结果

# In[10]:


def submit(result_df,filename='../data/submission.csv'):
    result_df=result_df.loc[:,['user_id','item_id']].drop_duplicates()
    print('结果共有：',len(result_df),'条数据')
    result_df.to_csv(filename,index=False)
    
result_df=user_df[(user_df.day=='2014-12-17')&(user_df.behavior_type==3)]
# 筛选出要预测的商品，因为我们只评估这部分商品
result_df=result_df[result_df.item_id.apply(lambda id:id in item_id_set)]
submit(result_df,'../data/submission1.csv')


# 6.将12月17日最后400条加入购物车的商品记录作为预测结果
# 
# 为什么选择400条，因为统计出来历史记录中，每天平均有400条购买记录

# In[11]:


o2o_user_df=user_df[user_df.item_id.apply(lambda id:id in item_id_set)]
buy_cnt=o2o_user_df[o2o_user_df.behavior_type==4].drop_duplicates(subset=['user_id','item_id','day']).day.value_counts()
pd.DataFrame(buy_cnt).plot()
plt.title('Buy Count Per Day')


# In[12]:


result_df=result_df.sort_values('hour',ascending=False).loc[:,['user_id','item_id']].drop_duplicates()
result_df=result_df.iloc[:400]
submit(result_df,'../data/submission2.csv')


# # 机器学习方法

# In[13]:


user_df=user_df[user_df.day>='2014-12-13']
o2o_user_df=o2o_user_df[o2o_user_df.day>='2014-12-13']
print('数据个数：',len(user_df))
print('与要预测商品相关的数据个数',len(o2o_user_df))


# In[14]:


def get_answer_dict(date):
    answer = user_df[(user_df.day==date)&(user_df.behavior_type==4)]
    answer = set(answer.apply(lambda item:'%s-%s'%(item.user_id,item.item_id),axis=1))
    return answer

def label_it(train_xs_df,target_date):
    answer=get_answer_dict(target_date)
    train_xs_df['label']=train_xs_df.apply(lambda item:1 if '%d-%d'%(item.user_id,item.item_id) in answer else 0,axis=1)
    return train_xs_df


# ## 抽取特征

# In[15]:


get_ipython().run_cell_magic('time', '', "\ndef get_features(target_date,user_df):\n    xs=[]\n    cnt=0\n    #target_date=datetime.datetime(2014,12,17)\n    start_date=target_date-datetime.timedelta(3)\n    tmp_df=user_df[(user_df.day>=start_date)&(user_df.day<target_date)]\n\n    for gid,items in tmp_df.groupby(by=['user_id','item_id']):\n        user_id,item_id=gid\n        x=[user_id,item_id]\n        vals=np.zeros([3,6,4])\n        for item in items.itertuples():\n            day=(target_date-item.day).days-1\n            hour=int(item.hour/4)\n            behavior=item.behavior_type-1\n            vals[day][hour][behavior]+=1\n        x.extend(list(vals.reshape((72))))\n        xs.append(x)\n        cnt+=1\n        if cnt%10000==0:\n            print(datetime.datetime.now(),'processed %d'%(cnt,))\n\n    headers=['user_id','item_id']\n    for i in range(3):\n        for j in range(6):\n            for k in range(4):\n                headers.append('d%d_h%d_b%d'%(i+1,j+1,k+1))\n    xs_df=pd.DataFrame(xs,columns=headers)\n    return xs_df")


# 商品在我们这里可分为o2o商品，和非o2o商品。我们只预测o2o商品的，但这部分数据正例个数太小，会使模型学习得不够充分，
# 所以我们在训练的时候，要所有的商品的数据（o2o商品和非o2o商品都用）。
# 
# 但对于验证集和测试集来说，我们只预测o2o商品就可以了

# In[55]:


get_ipython().run_cell_magic('time', '', "train_xs_df=get_features(datetime.datetime(2014,12,16),user_df)\nprint(datetime.datetime.now(),'train_xs_df processed')\n# 验证集和测试集只使用o2o的商品就可以了\nvalid_xs_df=get_features(datetime.datetime(2014,12,17),o2o_user_df)\nprint(datetime.datetime.now(),'valid_xs_df processed')\ntest_xs_df=get_features(datetime.datetime(2014,12,18),o2o_user_df)\nprint(datetime.datetime.now(),'test_xs_df processed')")


# In[56]:


label_it(train_xs_df,datetime.datetime(2014,12,16))
label_it(valid_xs_df,datetime.datetime(2014,12,17))


# In[58]:


positive_num=np.sum(train_xs_df.label)
negative_num=len(train_xs_df)-positive_num
print('正样本个数',positive_num,'负样本个数',negative_num,'负正样本比例',negative_num/positive_num)


# ## 对正样本进行过采样

# In[59]:



positive_xs_df=train_xs_df[train_xs_df.label==1]

positive_xs_df=positive_xs_df.sample(n=40000,replace=True)

sample_xs_df=pd.concat([train_xs_df,positive_xs_df])

sample_xs_df=sample_xs_df.sample(frac=1.0)


# In[60]:


from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import Normalizer
from sklearn.metrics import precision_recall_fscore_support,precision_score,recall_score,f1_score


# ## 对特征进行归一化处理

# In[61]:


scaler=Normalizer(norm='l1')
scaler.fit(sample_xs_df.drop(['user_id','item_id','label']))


# In[62]:


train_xs=scaler.transform(sample_xs_df.drop(['user_id','item_id','label'],axis=1))
valid_xs=scaler.transform(valid_xs_df.drop(['user_id','item_id','label'],axis=1))
test_xs=scaler.transform(test_xs_df.drop(['user_id','item_id'],axis=1))


# In[63]:


answer_cnt=len(o2o_user_df[(o2o_user_df.day=='2014-12-17')&(o2o_user_df.behavior_type==4)])
def evaluate(ytrue,ypred,answer_cnt):
    ypred=ypred>0.5
    right_cnt=np.sum(ytrue&ypred)
    predict_cnt=np.sum(ypred)
    precision=right_cnt/predict_cnt
    recall=right_cnt/answer_cnt
    f1=0
    if precision>0 or recall>0:
        f1=2*precision*recall/(precision+recall)
    print('预测数量',predict_cnt,'答案数量',answer_cnt)
    print('正确个数',right_cnt)
    print('precision',precision)
    print('recall',recall)
    print('f1',f1)
    return precision,recall,f1


# ## 逻辑回归模型

# In[38]:


clf=LogisticRegression(C=15)
#训练模型
clf.fit(train_xs,sample_xs_df.label)
#输出验证集结果
valid_yp=clf.predict(valid_xs)
#输出测试集结果
test_yp=clf.predict(test_xs)
#结果线下评估
evaluate(valid_xs_df.label,valid_yp,answer_cnt)
#测试集结果提交到文件中
test_xs_df['yp']=test_yp
submit(test_xs_df[test_xs_df.yp==1],filename='../data/submission_lr.csv')


# ## 梯度提升决策树

# In[72]:


get_ipython().run_cell_magic('time', '', "clf=GradientBoostingClassifier(n_estimators=250)\nclf.fit(train_xs,sample_xs_df.label)\n# 这里可以用predict也可以用predict_proba\n# predict只输出0和1，\n# predict_proba 可以输出概率值\nvalid_yp=clf.predict_proba(valid_xs)[:,1]\ntest_yp=clf.predict_proba(test_xs)[:,1]\nevaluate(valid_xs_df.label,valid_yp,answer_cnt)\ntest_xs_df['yp']=test_yp\nsubmit(test_xs_df[test_xs_df.yp>0.75],filename='../data/submission_gbdt.csv')")


# ## 随机森林模型

# In[40]:


clf=RandomForestClassifier(n_estimators=500)
clf.fit(train_xs,sample_xs_df.label)
# 这里可以用predict也可以用predict_proba
# predict只输出0和1，
# predict_proba 可以输出概率值
valid_yp=clf.predict_proba(valid_xs)[:,1]
test_yp=clf.predict_proba(test_xs)[:,1]
evaluate(valid_xs_df.label,valid_yp,answer_cnt)
test_xs_df['yp']=test_yp
submit(test_xs_df[test_xs_df.yp>0.5],filename='../data/submission_rf.csv')

