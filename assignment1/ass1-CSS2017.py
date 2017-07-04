
# coding: utf-8

# # Assignment 1
# 
# Please, answer the question by entering runable python code into the cells. Add comments at the beginning of each cell which list the packages that need to be installed (e.g., pip install collections). Run the code so that the output is visible in the notbook before you submit. The data files which you download should lie in the same directory as the python notebooks (use relative paths!). Use python 3.
# 
# 
# Submit the notebook (as .ipynb and .pdf) via email to clwagner@uni-koblenz.de until <font color="red">8.5.2017 (midnight CET)</font>.
# Subject of email: "CSS2017 ass 1"
# 
# Filename: firstname_lastname_ass1.ipynb 
# 
# 
# ## Analyzing Affiliation Networks and Social Networks (15 Points)
# 
# Download the following 2 datasets about the location-based social networking service Brightkite: http://snap.stanford.edu/data/loc-brightkite.html
# 
# One of the dataset contains all checkins between April 2008 and October 2010 (4.5 Mio checkins).
# 
# Beside the checkin data that constitute an affiliation network (two-mode network of users and locations), there is also a social network included  - i.e., a one-mode netowkr of users. It is an undirected one-mode network.
# 
# 

# In[1]:

import pandas as pd
import numpy as np
edges = pd.read_csv("Brightkite_edges.txt", sep="\t", names=['node1', 'node2'])
checkIns = pd.read_csv("Brightkite_totalCheckins.txt", sep="\t", names=['user', 'time', 'lat', 'lon', 'locId'])


# In[2]:

edges.sample(5)


# In[3]:

checkIns.sample(5)


# #### Location Popularity
# 
# Count the number of distinct brightkite users who checked-in at each location (using python!). What are the top 10 locations---i.e., the locations where most users checked in? Plot the rank ordered frequency distributions of locations (x-axis: locations ranked by the number of distinct users, y-axis: number of distinct users). <font color="red">(2 Points)</font>
# 

# In[4]:

grpLoc = checkIns.groupby('locId').user.nunique()


# In[5]:

grpLoc.sort_values(ascending=False, inplace=True)
grpLoc.head(11)


# In[35]:

import matplotlib.pyplot as plt
plt.plot([i for i in range(1,len(grpLoc)+1)], list(grpLoc), 'b-')
plt.title('Location popularity')
plt.ylabel('Number of users in a location')
plt.xlabel('Location rank')
plt.xscale('log')
plt.axis([0, 1000000, 0, 3300])
plt.show()


# #### Fold Location-User Network
# 
# Construct an unweighted and undirected two-mode network of brightkite users and locations. A user and a location are connected if the user checked-in at the location at least once. Compute the degree of each location in the two-mode network and list the top 10 location-ids with their corresponding degree. Make a plot that shows the ranked locations (x-axis) and their degree (y-axis).
# 
# Fold the two-mode network and construct a location network. Since folding the full network is very slow, select only locations where at least 10 users checked in.
# 
# What are the most central locations in the one-mode network and what does that mean? 
# Use different centrality measures and order nodes by these centralities and print the top 10 nodes for each measure. Discuss for each centrality measure what it does and how you can interpret the results. For example, what does it mean that location X has the highest betweeness centrality? How is location X different from location Y which has the highest degree centrality?
# <font color="red">(3 Points)</font>

# In[36]:

checkIns.shape


# In[37]:

checkIns[checkIns.duplicated(subset=['user', 'locId'], keep='first')].shape


# In[38]:

import networkx as nx
locUsrNtwrk = nx.from_pandas_dataframe(checkIns[['user', 'locId']], source='user', target='locId')


# In[39]:

nx.number_of_edges(locUsrNtwrk)


# In[40]:

nx.is_bipartite(locUsrNtwrk)


# In[41]:

nx.is_connected(locUsrNtwrk)


# In[42]:

nx.is_directed(locUsrNtwrk)


# In[43]:

degrees = locUsrNtwrk.degree(checkIns['locId'].unique())


# In[44]:

degreesSorted = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
print(degreesSorted[:10])


# In[45]:

degreesSortedList = [i[1] for i in degreesSorted]


# In[48]:

plt.plot([i for i in range(1, len(degreesSortedList)+1)],degreesSortedList)
plt.title('Location popularity')
plt.ylabel('Number of users in a location')
plt.xlabel('Location rank')
plt.xscale('log')
plt.axis([0, 1000000, 0, 3400])
plt.show()


# In[ ]:

overTen = [i[0] for i in degreesSorted if i[1]>=150]


# In[ ]:

foldedLoc = nx.projected_graph(locUsrNtwrk, overTen)


# In[ ]:

nx.number_of_nodes(foldedLoc)


# Betweenness centrality

# In[ ]:

bet_cent = nx.betweenness_centrality(foldedLoc, k=5)


# In[ ]:

bet_cent_sorted = sorted(bet_cent.items(), key=lambda x: x[1], reverse=True)
print(bet_cent_sorted[:10])


# Degree centrality

# In[ ]:

deg_cent = nx.degree_centrality(foldedLoc)


# In[ ]:

deg_cent_sorted = sorted(deg_cent.items(), key=lambda x: x[1], reverse=True)
print(deg_cent_sorted[:10])


# Closeness centrality

# In[ ]:

close_cent = nx.closeness_centrality(foldedLoc)
close_cent_sorted = sorted(close_cent.items(), key=lambda x: x[1], reverse=True)
print(close_cent_sorted[:10])


# In[ ]:

#TODO explain centrality measures


# #### Entropy of Location Popularity Distribution
# 
# How evenly distributed is the attention of brightkite users towards locations? The attention a location receives is measured by the number of distinct users who checked in at this loction. Compute the normalized entropy of the degree distribution. The degree of a location corresponds to the number of distinct users who checked in there.
# 
# To that end, you want to have one vector where each element corresponds to a distinct location. The values for each element are then determined by the number of distinct users that have checked-in at that location. Then, calculate normalized entropy for this vector.
# 
# Assume we ignore all locations where 0-1 different people checked in. That means we make the long tail of the degree distribution shorter. How would the entropy change? <font color="red">(3 Points)</font>

# In[ ]:

#TODO degreesSortedList - entropy
from scipy.stats import entropy
from math import log
print(entropy(degreesSortedList))

#TODO entropy with 0-1 people ignored
degreesSortedListOverOne = [i[1] for i in degreesSorted if i[1]>1]
print(entropy(degreesSortedListOverOne))
# Maybe divide with log2(n)


# #### Describe Location Popularity Distribution
# 
# In how many distinct locations did users check in on average in brightkite? 
# Create a boxplot that shows the distribution of the number of distinct checkin-locations per user. What is the mean and the variance of this distribution? Compute X for the following statement: Half of the users checked in in more than X locations?  <font color="red">(2 Points)</font>

# In[ ]:

grpUsr = checkIns.groupby('user')['locId'].nunique()


# In[ ]:

stats = grpUsr.describe()
print("Mean of this distribution is at {}, and the variance is {}. \nHalf of the users checked in more than {} locations."
      .format(stats['mean'], stats['std']**2, stats['50%']))


# In[ ]:

plt.boxplot(grpUsr, 0 , '', 0)
plt.show()


# #### Describe Social Network
# 
# Load the full Brightkite social network (loc-brightkite_edges.txt.gz). Describe the network.
# 
# Some ideas on what you could report: What's the number of nodes, edges and average degree of a node?  What is the size of the largest maximal clique in the network? How well connected is the network? <font color="red">(2 Points)</font>

# In[ ]:

br_network = nx.from_pandas_dataframe(edges, source='node1', target='node2')


# In[ ]:

br_network.number_of_edges()


# In[ ]:

br_network.number_of_nodes()


# In[ ]:

br_degrees = nx.degree(br_network)
np.mean(list(br_degrees.values()))


# In[ ]:

cliq_num = nx.graph_clique_number(br_network)


# In[ ]:

cliq_num


# In[ ]:

#TODO How well is the network connected?


# #### Draw Social Network
# 
# Select the top location (i.e., the location where most distinct brightkite user checked in) and filter the brightkite social network (i.e., Brightkite_edges.txt) so that it only contains users (nodes) which checked in at the top location.
# Load this subpart of the social networks into python (using the NetworkX library)
# Plot the sub-network in an appealing way. <font color="red">(3 Points)</font>

# In[ ]:

topLoc = grpLoc.index[1]
been2TopLoc = checkIns[checkIns['locId']==topLoc].user.unique()


# In[ ]:

viz_edges = edges[edges['node1'].isin(been2TopLoc)].append(edges[edges['node2'].isin(been2TopLoc)])


# In[ ]:

viz_edges = viz_edges.drop_duplicates()


# In[ ]:

viz_net = nx.from_pandas_dataframe(viz_edges, source='node1', target='node2')


# In[ ]:

nx.draw(viz_net)
plt.savefig("simple_path.png")


# In[ ]:

print("finished?")

