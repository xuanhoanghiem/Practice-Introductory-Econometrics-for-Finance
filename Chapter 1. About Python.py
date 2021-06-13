#!/usr/bin/env python
# coding: utf-8

# In[49]:


import numpy as np # Load the library
a=np.linspace(-np.pi, np.pi, 100) # Create even grids from -π to π
b=np.cos(a) # Apply cosine to each element of a
c=np.sin(a) # Apply sin to each element of a


# In[50]:


b@c # Take the inner product


# In[51]:


# The Scipy library is built on top of NumPy and provides additional functionality.
# We can calculate the pdf 
from scipy.stats import norm
from scipy.integrate import quad
ϕ=norm()
value, error=quad(ϕ.pdf, -2, 2) # Here we integrate using Gaussian quadrature 
value
# The Scipy library includes many of the standard routines in:
    # Linear algebra
    # Integration
    # Interpolation
    # Optimization
    # Distributions and random number generation
    # Signal Processing


# In[52]:


# 1.4.2. Graphics
# The most popular and comprehensive Python library for creating figures and graphs is Matplotlib with functionality including:
    # Plots, histograms, contour images, 3D graphs, bar charts...
    # Output in many formats (PDF, PNG, EPS...)
    # LaTeX integration
# Other graphic libraries include: Ploty; Bokeh; VPython (for 3D graphics and animations)


# In[53]:


# Symbolic Algebra
# It is useful to be able to manipulate symbolic expressions as in Mathematica or Maple
# The SymPy library provides this functionality from within the Python shell.
from sympy import Symbol
x, y = Symbol('x'), Symbol('y') # This command treats 'x' and 'y' as algebraic symbols.
x + x + x - y


# In[54]:


expression1=(x+y)**2
expression1.expand()


# In[55]:


expression2=(x-y)**4
expression2.expand()


# In[56]:


# We can also use Scipy to solve polynomials
from sympy import solve
solve(x**2 + x + 2)


# In[57]:


solve(x**2 + 4*x +4)


# In[58]:


# We can use Sympy to calculate limits, derivatives and integrals
from sympy import limit, sin, diff
limit(1/x, x, 0)


# In[59]:


limit(sin(x)/x, x, 0)


# In[60]:


diff(sin(x), x)


# In[61]:


diff(cos(x), x)


# In[ ]:


# 1.4.4. Statistics
import pandas as pd
np.random.seed(1234)
data=np.random.randn(5,2) # 5x2 matrix of N(0,1) random draws.
dates=pd.date_range('28/12/2010', periods=5)
df=pd.DataFrame(data, columns = ('price', 'weight'), index=dates)
print(df)


# In[ ]:


df.mean()


# In[ ]:


# Other useful statistical libraries include:
    # statsmodels - which includes various statistical routines.
    # scikit-learn - machine learning in Python
    # pyMC = for Bayesian data analysis
    # pystan Bayesian analysis based on stan


# In[62]:


# 1.4.5. Networks and Graphs
# Python has many libraries for studying graphs. 
# A well-known example is NetworkX which has features:
    # Standard graph algorithms for analyzing networks.
    # Plotting routines. 
import networkx as nx
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(1234)


# Generate a random graph
p=dict((i, (np.random.uniform(0,1), np.random.uniform(0,1)))
      for i in range(200))
g=nx.random_geometric_graph(200, 0.12, pos=p)
pos=nx.get_node_attributes(g, 'pos')


# Find node nearest the center point (0.5, 0.5)
dists=[(x-0.5)**2 + (y-0.5)**2 for x, y in list(pos.values)]
ncenter=np.argmin(dists)

# Plot graph, coloring by path length from central node
p=nx.single_source_shortest_path_length(g, ncenter)
plt.figure()
nx.draw_networkx_edges(g, pos, alpha=0.4)
nx.draw_networkx_nodes(g, 
                      pos, 
                      nodelist=list(p.keys()), 
                      node_size=120, alpha=0.5, 
                      node_color=list(p.values()),
                      cmap=plt.cm.jet_r)
plt.show()


# In[71]:


import matplotlib
matplotlib.__version__ # how to check the version of matplotlib


# In[69]:


import numpy
print(numpy.version.version) # How to check for version of NumPy


# In[74]:


numpy.__version__


# In[77]:


pip list


# In[78]:


pip show sympy


# In[ ]:




