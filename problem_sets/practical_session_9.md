# Graph Data Analysis

## Introduction

In this problem set, you will play with the Paris metro line. You may find in ```./data/paris_metro_line.txt``` a file structured as follows,

```
line 1,station_11,station_12,station_13,station_14
line 2,station_21,station_22,station_23
```

A second file, called ```./data/metro_coordinates.csv``` contains pair (latitude, longitude), in degrees, for all the metro stations. Read the csv file, then create two new columns called $x$ and $y$ based on the formulas,

$$x = R (\psi - \psi_{0})$$

and

$$y = R \log ( tan(\pi / 4 + \phi / 2) )$$

where $R = 6371$, $\phi$ is the latitude, $\psi$ is the longitude and $\phi_{0} = 2.347324$ is __the longitude of Châtelet-Les Halles__ (why?). These formulas come from the Mercator projection. The value for $R$ corresponds to the Earth radius (in km). 

## Data Cleaning

Your first step will be formatting the data to fit the format of a graph. Based on the .txt file, create a Pandas DataFrame with 3 columns: Source, Target and Metro Line. The Source corresponds to the departure station, whereas the Target corresponds to the arrival station. For instance, a possible line in this DataFrame would be,

```
Louvre Rivoli,Châtelet,Line 1
```

__Note.__ Do not worry about directions. We assume that (Source $\iff$ Target) for simplicity.

## Modeling

### NetworkX

Convert your DataFrame into a NetworkX Graph object using the following steps,

- Find the unique station names, then add these as nodes in a graph $G$.
- For each line in your DataFrame, add an edge between the two stations.

### Simulating diffusion of passangers over the metro-line

In this problem, you will create a simulation based on the heat equation saw in the lecture. Your simulation should start with 1000 people on 6 random different stations. You will then iterate the Laplace equation

__Hint.__ Show the function over $G$ in logarithmic scale.

__Conceptual Questions.__

- As time passes, what happens to the distributino of people over the Metro?
- What are the limitations of this simulation?