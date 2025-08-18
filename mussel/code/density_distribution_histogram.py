import matplotlib.pyplot as plt
import pandas as pd
# Draw a density distribution histogram of the movement ratio of mussel 
# Plot settings
plt.rcParams['font.family'] = 'Arial'
plt.rcParams["font.size"] = 16
# Parse the CSV data
mussel_path ='D:/khaki/ultralytics-8.3.27/mussel/results/move_dis_and_rate-2017-2018-revise.csv'
df =pd.read_csv(mussel_path)
fig, ax = plt.subplots(figsize=(16, 4.5))
plt.hist(df["rate"], # plot data
        bins = 20, # Specify the group spacing
        density =True, # Set as frequency histogram
        #cumulative = True, # Accumulate histogram
        color = '#f2a9a2ff', # Specify fill color
        edgecolor = 'w', # Specify the boundary color of the histogram
        #label = "Frequency"# Label
        )

# Set the coordinate axis labels and titles
#plt.title('Frequency histogram',fontsize = 20)
plt.xlabel('Movement Ratio')
plt.ylabel('Probability Density')

# Display the legend
plt.grid()
plt.legend(loc = 'best')
# Display the graph
plt.show()