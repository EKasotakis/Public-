import pandas
import cdt
import networkx as nx
import matplotlib.pyplot as plt
# load data
df = pandas.read_csv("a1_data.csv")

# Get skeleton graph
# initialize graph lasso
glasso = cdt.independence.graph.Glasso()
# apply graph lasso to data
skeleton = glasso.predict(df)
# visualize network
fig = plt.figure(figsize=(15,10))
nx.draw_networkx(skeleton, font_size=18, font_color='r')
plt.title('Skeleton graph')
plt.savefig('skeleton.png')

# Use causal discovery to get causal models
# PC algorithm
model_pc = cdt.causality.graph.PC()
graph_pc = model_pc.predict(df, skeleton)

# visualize network
fig=plt.figure(figsize=(15,10))
nx.draw_networkx(graph_pc, font_size=18, font_color='r')
plt.title('Causal model')
plt.savefig('causal_model.png')

