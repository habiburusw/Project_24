# Customer_Segmentation
This is a Customer Segmentation model made in Python 

## Goal

The goal of the first part of this project is to create segments of wholesale customers and then try to characterize those segments based on their content. 
For the clusters-segments to be created multiple clustering methods were used.

## The dataset

The dataset used contains 440 wholesale customers and their spendings on the following goods: Fresh, Milk, Grocery, Frozen, Detergents_Paper and Delicassen. 
Also, there are two more features in the dataset: Channel and Region. 
All the spendings in products are counted in dollars. 
The dataset is free of missing values.

*Figure 1: Dataset representation*

![image](https://user-images.githubusercontent.com/82097084/165760684-44948037-798e-472a-8e47-623b47e8a3a7.png)

The following visualization offers a better understanding on how the different features are related with one another:

*Figure 2: Dataset Visualization*

![image](https://user-images.githubusercontent.com/82097084/165761254-6005a01e-d475-4888-b719-5bf2c2b54bee.png)

**Preprocessing Phase**

The preprocessing phase contains the following steps:
-	Removing features Channel and Region
-	Scale data
-	Reduce dimensions
The decision to remove features Channel and Region was made since there was no information on what those represent. 
This lack of information could be misleading and distorting for the analysis to follow. 
As a second step, the data were scaled based on logarithm with base 10 to give all values equal weighting and produce trustworthy and unbiased clusters.
The last step was to reduce the dimensions of the dataset using PCA. The decision was based on the following graph:

*Figure 3: Explained variance ratio – Number of Components*

![image](https://user-images.githubusercontent.com/82097084/165761395-05f38a4a-d976-43ae-b827-c9d8a7729337.png)

As one can observe in Figure 3, 4 dimensions still explain about the 94% of the variance of the dataset. Also, fewer dimensions are easier to handle.

## Unsupervised Machine Learning (Clustering)

To create cluster-segments out of the dataset multiple clustering techniques were used:
-	K – Means Clustering
-	Agglomerative Clustering
-	Gaussian Mixture Clustering
-	DBSCAN Clustering

**K – Means Clustering**

The first technique used to segment the customers was K – Means. 
To decide the optimal number of clusters – segments of costumers to create, a combination of two metrics was used: 
Sum of Squares for Errors (SSE) and Silhouette Coefficients. 

Both metrics were plotted against the number of clusters and the results were the following:

*Figure 4: SSE values – Number of Clusters (Elbow Method)*

![image](https://user-images.githubusercontent.com/82097084/165761661-fc6ae108-d7e2-4923-82e2-e166284f8260.png)

*Figure 5: Silhouette Coefficients – Number of Clusters*

![image](https://user-images.githubusercontent.com/82097084/165761727-d4dcc301-95d2-474a-a37d-e1ed4b420ac3.png)

Based on those graphs, the best combination of SSE and Silhouette Score seems to be realized at three (3) clusters.
-SSE: 2986
-Silhouette Score: 0.28

After deciding the number of clusters to be created, the clusters were formed with the K – Means algorithm. 
To be able to understand what kind of customers each cluster contains and characterize those segments, box - plots were used. 

*Figure 6: Box – plots for Cluster #0*

![image](https://user-images.githubusercontent.com/82097084/165761865-85d92d99-a407-4bb9-989a-c3e0c93818da.png)

The first cluster (Figure 6) created seems to contain customers that on average spend more on Fresh, Grocery and Milk products while their spending on other categories is relatively low.

*Figure 7: Box – plots for Cluster #1*

![image](https://user-images.githubusercontent.com/82097084/165761977-f1b8dad4-d2c6-4ac0-b137-a79928de5d9b.png)


The second cluster (Figure 7) created contains costumers that spend a lot of money on Fresh products but overall, their spendings are low.
The costumers of this cluster, on average, spend small amounts of money on goods, except from Fresh products.

*Figure 8: Box – plots for Cluster #2*

![image](https://user-images.githubusercontent.com/82097084/165762158-128a6d6a-7ef4-4e36-bfa1-9815b059e7c5.png)


The third cluster (Figure 8) contains costumers that spend a lot of money on Grocery, Milk, and Detergent Paper while their spendings on the other products are relatively low. 
The customers of this cluster, in general, have a more balanced spending routine.
Overall, the costumers of the different clusters have similar spending routines on some products (for instance Cluster #0 seems to spend a lot on Fresh while the same happens for the costumers of Cluster #1) 
but at the same time they are differentiated on their spendings on other products. So, the clusters created could offer valuable insights on the different behaviors.

**Agglomerative Clustering**

The second technique used to segment customers was Agglomerative Clustering.
To decide the optimal number of clusters – segments needed to be created the Silhouette Coefficients metric was used. The following plot was produced:

*Figure 9: Silhouette Coefficients – Number of Clusters*

![image](https://user-images.githubusercontent.com/82097084/165762433-906cdb68-08d1-438a-b1d3-a4052703c230.png)

Based on this graph, the optimal number of clusters seems to be 6 (with minimum difference in Silhouette Score in comparison to 3 clusters).
-Silhouette Score: 0.205

Then the clusters were created with the Agglomerative Clustering algorithm.
For the better understanding of the contents of the clusters, box – plots were used.

*Figure 10: Box – plots for Cluster #0*

![image](https://user-images.githubusercontent.com/82097084/165762568-cb26415d-212a-4ca1-a735-95e659e7b755.png)

The first cluster (Figure 10) created by this clustering technique seems to contain costumers with equally distributed spendings overall. 
Based on the box – plots, one can observe that they tend to spend more on Grocery, Fresh and Milk products.

*Figure 11: Box – plots for Cluster #1*

![image](https://user-images.githubusercontent.com/82097084/165762732-2027c9cd-e11e-4133-a3ce-c7a29b8e701c.png)

The second cluster (Figure 11) created seems to contain customers that spend low amounts of money on the available (in the dataset) products. 
Also, their spendings are almost equally distributed between the goods

*Figure 12: Box – plots for Cluster #2*

![image](https://user-images.githubusercontent.com/82097084/165762819-694e5ad4-ff0b-403e-9f76-34885b7f4e9e.png)

The third cluster (Figure 12) created seems to spend a lot of money on Fresh products, while their spendings on the other goods are average to low. 

*Figure 13: Box – plots for Cluster #3*

![image](https://user-images.githubusercontent.com/82097084/165762920-4abf8fb9-768e-430b-be13-782745f5f87f.png)

The fourth cluster (Figure 13) created seems to contain customers that spend a lot on these three categories: Grocery, Milk and Detergents paper with Grocery leading followed by the other two goods. 
Their spending on other goods are minimum. 

*Figure 14: Box – plots for Cluster #4*

![image](https://user-images.githubusercontent.com/82097084/165763046-f245ec19-b873-4ae5-866b-18203aa68e8b.png)

The costumers of the fifth cluster (Figure 14) spent large amounts of money on Fresh products while their spendings on other products are minimum. 

*Figure 15: Box – plots for Cluster #5*

![image](https://user-images.githubusercontent.com/82097084/165763141-ef66bffc-f0f9-467d-a0d6-b7d6930f759c.png)

The sixth cluster (Figure 15) seems to contain customers that spend a lot on Fresh products while their spendings on the other goods are relatively small. 
Overall, the customers of clusters #2, #4 and #5 seem to have some common spending patterns. 
They spend a lot on Fresh while their spending on the other goods is close. Cluster #1 contains customers that spend minimum amounts of money on the available products. 
Cluster #0 customers have equal spending on average on all the categories.  
Cluster #3 contains wholesale customers that spend almost exclusively in some categories.

**Gaussian Mixture Clustering**

The third technique used to segment customers was Gaussian Mixture Clustering.
In order to decide how many clusters should be created the following plots were produced:

*Figure 16: Silhouette Coefficients – Number of Clusters*

![image](https://user-images.githubusercontent.com/82097084/165763336-86ed1d89-a713-4afe-a5d4-c1069ff39c06.png)

*Figure 17: log-likelihood – Number of Clusters*

![image](https://user-images.githubusercontent.com/82097084/165763390-1675443a-6613-4336-94d7-4e157e01db86.png)

For 3 clusters the results were the following:
- Silhouette Score: 0.26
- Log-likelihood: -6.86

Based on those diagrams the decision to create 3 clusters was made.
Then, the clusters were created with the Gaussian Mixture algorithm.

*Figure 18: Box – plots for Cluster #0*

![image](https://user-images.githubusercontent.com/82097084/165763496-8c66c6cf-17bb-4b39-8979-76580c5dfdfb.png)

The first cluster (Figure 18) created contains customers that spend a lot on Fresh products while they tend to spend low amounts of money on the other products.

*Figure 19: Box – plots for Cluster #1*

![image](https://user-images.githubusercontent.com/82097084/165763576-87903999-465e-4293-858e-4760647b395d.png)

The second cluster (Figure 19) seems to contain customers that have some uniformity in their spending in the categories Grocery, Fresh and Milk where they seem to spend large amounts of money in comparison to the other categories.

*Figure 20: Box – plots for Cluster #2*

![image](https://user-images.githubusercontent.com/82097084/165763987-b0d83577-bc37-4799-9851-52ebd4fd0f30.png)

On average, Cluster #2 (Figure 20) contains costumers that spend more on Grocery and Milk, but without spending huge amounts of money.
Overall, the clusters created with this method cannot be easily characterized based on how much they spend because they spend on average similar amounts of money on products. 
So, there is no obvious way they can be grouped in cluster groups. We could point out that Cluster #0 and Cluster #2 spend sufficiently more on some products (#0 on Fresh and #2 on Grocery and Milk) while Cluster #1 has somehow similar spendings between products.

**DBSCAN Clustering**

The last clustering method used in the process of creating costumer segments was DBSCAN Clustering.
At this stage, it was essential to find the best parameters possible that should be provided to the algorithm. 
The following plot was created to offer some insights:

*Figure 21: Distance – Points*

![image](https://user-images.githubusercontent.com/82097084/165764213-1d465a26-2533-4616-8f1a-ba8a45b349d7.png)

By finding the elbow point in Figure 21 the decision to set eps hyper-parameter to 1.2 was made. 
Also, to tune the minimum number of points hyper-parameter the trial-and-error method was used and finally it was set to be 7 points.
The result of these decisions was the creation of 3 clusters by the DBSCAN algorithm. 

*Figure 22: Box – plots for Cluster #-1*

![image](https://user-images.githubusercontent.com/82097084/165764302-933c3928-e64c-46c9-9430-bc3719a093e9.png)

The first cluster (Figure 22) created by the DBSCAN algorithm seems to contain customers that on average spent average to low amounts of money on products with an exception in Grocery products. 
Spendings on Delicassen, Frozen are minimum.

*Figure 23: Box – plots for Cluster #0*

![image](https://user-images.githubusercontent.com/82097084/165764406-2f2396cd-d6f4-4329-8e05-d9b5730d8031.png)

The second cluster (Figure 23) created, seems to contain costumers with high spendings on Fresh products while spendings on other goods are average to low.

*Figure 24: Box – plots for Cluster #1*

![image](https://user-images.githubusercontent.com/82097084/165764596-4b9b3cb3-b8dc-49b6-9771-fd35f0690f23.png)

The third cluster (Figure 24) created contains customers with high spendings on Grocery and relatively high spendings on Detergent Paper and Milk products while spendings on the other goods are low to minimum.
Overall, from this method it is not possible to group clusters together. Cluster #1 seems to be differentiated because of the relative big spendings on some products.
While Cluster #-1 and #0 cannot be differentiated easily from each other, Cluster #-1 contains, on average, bigger spenders in comparison to Cluster #0. 

## INSTRUCTIONS:

The user of the code names 'Clustering' should change the path and replace it with a valid one for his path of the dataset. 

The code is running just by clicking run.

Tree classifiers are going to produce some word files in the file of the code and they can be copy and pasted in the following site http://www.webgraphviz.com/ for best visualization.

