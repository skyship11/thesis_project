import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

'''
# convert emotion txt file to csv file
column_names = ['image', 'emotion']
df = pd.read_csv('EmoLabel/list_patition_label.txt', delim_whitespace=True, names=column_names)
df.to_csv('emotion.csv', index=False)
'''

'''
# Define the directory where the txt files are stored
directory = 'manual/'

# Initialize a list to hold the extracted data
extracted_data = []

# Loop over the files in the directory
for filename in sorted(os.listdir(directory)):
    if filename.endswith(".txt"):
        # Construct the full file path
        file_path = os.path.join(directory, filename)
        # Read the last three lines of the file
        with open(file_path, 'r') as file:
            lines = file.readlines()[-3:]  # Get the last three lines
            # Check if we have exactly three lines for gender, race, age
            if len(lines) == 3:
                # Extract the gender, race, and age information
                gender = lines[0].strip()
                race = lines[1].strip()
                age = lines[2].strip()
                # Append the extracted data to the list
                extracted_data.append([gender, race, age])

# Create a DataFrame from the extracted data
df_attributes = pd.DataFrame(extracted_data, columns=['gender', 'race', 'age'])
df_attributes.to_csv('attributes_luan.csv', index=False)
'''

'''
# Load the CSV file using pandas
csv_file_path = 'attributes_luan.csv'
df = pd.read_csv(csv_file_path)

# Split the DataFrame into two parts
first_part = df[:3068]
second_part = df[3068:]

# Concatenate the second part with the first part
rearranged_df = pd.concat([second_part, first_part], ignore_index=True)

# Save the rearranged DataFrame to a new CSV file
rearranged_csv_path = 'attributes.csv'
rearranged_df.to_csv(rearranged_csv_path, index=False)
'''

'''
# Load both CSV files using pandas
attributes_csv_path = 'attributes.csv'
emotion_csv_path = 'emotion.csv'

# Read the CSV files into DataFrames
attributes_df = pd.read_csv(attributes_csv_path)
emotion_df = pd.read_csv(emotion_csv_path)

# Assuming both DataFrames have the same order and number of rows and can be joined row-wise
# We will concatenate them horizontally (side by side)
combined_df = pd.concat([emotion_df, attributes_df], axis=1)

# Save the combined DataFrame to a new CSV file
combined_csv_path = 'total_attributes.csv'
combined_df.to_csv(combined_csv_path, index=False)
'''

'''
file_path = 'rafdb.csv'
data = pd.read_csv(file_path)
data['Image'] = data['Image'].apply(lambda x: x.replace('.jpg', '_aligned.jpg'))

new_file_path = 'rafdb_aligned.csv'
data.to_csv(new_file_path, index=False)

train_data = data[data['Image'].str.contains('train')]
test_data = data[data['Image'].str.contains('test')]

# print('train data number: ', train_data.shape[0])     # 11519
# print('test data number: ', test_data.shape[0])       # 2869

base_path = 'aligned'
train_path = os.path.join(base_path, 'train')
test_path = os.path.join(base_path, 'test')

os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# move training images
for image_name in train_data['Image']:
    source = os.path.join(base_path, image_name)
    destination = os.path.join(train_path, image_name)
    shutil.move(source, destination)

# move test images
for image_name in test_data['Image']:
    source = os.path.join(base_path, image_name)
    destination = os.path.join(test_path, image_name)
    shutil.move(source, destination)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Sort images into corresponding Emotion subfolders
def sort_images_to_folders(data, base_dir):
    for index, row in data.iterrows():
        source = os.path.join(base_dir, row['Image'])
        emotion_folder = os.path.join(base_dir, str(row['Emotion']))
        ensure_dir(emotion_folder)

        # build target path
        destination = os.path.join(emotion_folder, row['Image'].split('/')[-1])

        # move files
        if os.path.exists(source):
            shutil.move(source, destination)
        else:
            print(f"File not found: {source}")

sort_images_to_folders(data[data['Image'].str.contains('train')], train_path)
sort_images_to_folders(data[data['Image'].str.contains('test')], test_path)
'''

# Plot settings
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
#
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)
#
fig_size=(7,6)
title_font = {'size':BIGGER_SIZE, 'color':'black', 'weight':'normal'} # Bottom vertical alignment for more space


'''
Plot the 4 figures about emotion, gender, race, and age distributions
'''
# Load the CSV file into a DataFrame
rafdb_data = pd.read_csv('rafdb_aligned.csv')

# Define the mappings for each category
emotion_map = {
    1: 'Surprise',
    2: 'Fear',
    3: 'Disgust',
    4: 'Happiness',
    5: 'Sadness',
    6: 'Anger',
    7: 'Neutral'
}
# remove the unsure gender
gender_map = {
    0: 'Male',
    1: 'Female'
}
race_map = {
    0: 'Caucasian',
    1: 'African-American',
    2: 'Asian'
}
age_map = {
    0: '0-3',
    1: '4-19',
    2: '20-39',
    3: '40-69',
    4: '70+'
}
mapping_list = [emotion_map, gender_map, race_map, age_map]

# Function to calculate percentage counts
def attribute_counts(attribute, mapping_dict):
    counts = rafdb_data[attribute].value_counts(normalize=True, sort=False) * 100  # Convert counts to percentage
    counts = counts.reset_index()
    counts.columns = [attribute, 'Percentage']
    counts[attribute] = counts[attribute].map(mapping_dict)
    order = [mapping_dict[i] for i in sorted(mapping_dict.keys())]
    counts[attribute] = pd.Categorical(counts[attribute], categories=order, ordered=True)
    counts = counts.sort_values(attribute)

    return counts

# Define a function to plot bar graphs for categorical data
def plot_distributions(df_list, col_list, title_list):
    fig, axes = plt.subplots(2, 2, figsize=fig_size)
    axes = axes.flatten()

    for i, (df, col, title) in enumerate(zip(df_list, col_list, title_list)):
        ax = axes[i]
        sns.barplot(x=col, y='Percentage', data=df, ax=ax)
        ax.set_title(title, fontdict=title_font)
        ax.set_xlabel('')
        ax.set_ylabel('Percentage (%)')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig('statistic_figure/rafdb_attributes_distribution.pdf')
    plt.show()

# Read only the headers
headers_df = pd.read_csv('rafdb_aligned.csv', nrows=0)
headers = headers_df.columns.tolist()[1:]

counts_list = []
for i in range(4):
    counts = attribute_counts(headers[i], mapping_list[i])
    counts_list.append(counts)

plot_distributions(counts_list, headers, headers)


'''
try to find the number of different categories according to attributes
'''

# Calculate the total number of data points
total_data_points = rafdb_data.shape[0]
headers = headers_df.columns.tolist()[2:]

def mix_count(attribute):
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('the number of '+attribute+' and emotion')
    # Perform a groupby operation to count the number of emotions per gender
    emotion_count_per_attribute = rafdb_data.groupby([attribute, 'emotion']).size().reset_index(name='count')

    return emotion_count_per_attribute

# the number of each class for different attributes
def per_attribute(attribute):
    print('*******************************')
    unique_attribute = rafdb_data[attribute].unique()
    num_unique = len(unique_attribute)

    for i in range(num_unique):
        category = rafdb_data[rafdb_data[attribute] == i]
        print(f"{attribute} {i}: {(category.shape[0] / total_data_points) * 100:.1f}")

# The percentage of each sensitive attribute
for i in headers:
    print(mix_count(i))
    print(per_attribute(i))

# The percentage of each emotion
for i in range(1, 8):
    emotion = rafdb_data[rafdb_data['emotion'] == i]
    print(f"emotion {i}: {(emotion.shape[0] / total_data_points) * 100:.1f}")



'''
Plot the bar figures that show the number of combined attributes based on one sensitive attribute
eg. x: gender, y: number, height of bar is the number of combined race and age based on male
'''
# similar grouped bars in one place
def custom_sort_gender(comb, secondary_labels):
    age_priority = {'0-3': 0, '4-19': 1, '20-39': 2, '40-69': 3, '70+': 4}
    race_priority = {'Caucasian': 0, 'African-American': 1, 'Asian': 2}
    return (age_priority[secondary_labels['age'][comb[1]]], race_priority[secondary_labels['race'][comb[0]]])

def custom_sort_race(comb, secondary_labels):
    gender_priority = {'Male': 0, 'Female': 1}
    age_priority = {'0-3': 0, '4-19': 1, '20-39': 2, '40-69': 3, '70+': 4}
    return (age_priority[secondary_labels['age'][comb[1]]], gender_priority[secondary_labels['gender'][comb[0]]])

def custom_sort_age(comb, secondary_labels):
    gender_priority = {'Male': 0, 'Female': 1}
    race_priority = {'Caucasian': 0, 'African-American': 1, 'Asian': 2}
    return (race_priority[secondary_labels['race'][comb[1]]], gender_priority[secondary_labels['gender'][comb[0]]])

def plot_grouped_bars_with_age_gaps(data, primary_attribute, secondary_attributes, primary_labels, secondary_labels, sort_function):
    for primary_key, primary_name in primary_labels.items():
        primary_data = data[data[primary_attribute] == primary_key]

        combinations = primary_data[secondary_attributes].drop_duplicates().sort_values(by=secondary_attributes).values

        # Apply custom sort
        combinations = sorted(combinations, key=lambda comb: sort_function(comb, secondary_labels))

        color_map = plt.cm.get_cmap('viridis', len(combinations))
        colors = [color_map(i) for i in range(len(combinations))]
        legend_labels = [' & '.join([secondary_labels[attr][val] for attr, val in zip(secondary_attributes, comb)]) for comb in combinations]

        fig, ax = plt.subplots(figsize=fig_size)
        width = 0.5
        positions = []
        gap = 1.0  # Larger gap between groups for clarity
        current_position = 0

        last_group = None
        total_counts = primary_data.shape[0]

        for idx, (comb, color, label) in enumerate(zip(combinations, colors, legend_labels)):
            current_group = sort_function(comb, secondary_labels)[0]
            if last_group is not None and last_group != current_group:
                current_position += gap
            positions.append(current_position)
            last_group = current_group

            mask = True
            for attr, val in zip(secondary_attributes, comb):
                mask &= (primary_data[attr] == val)
            filtered_data = primary_data[mask]
            count = filtered_data.shape[0]
            percentage = (count / total_counts) * 100 if total_counts > 0 else 0

            ax.bar(current_position, percentage, width, color=color, label=label)
            current_position += 0.5

        ax.set_title(f'Percentages for {primary_attribute} ({primary_name}) with grouped attributes', fontdict=title_font)
        ax.set_ylabel('Percentage (%)')
        ax.set_xticks(positions)
        ax.set_xticklabels(legend_labels, rotation=45, ha="right")
        # ax.legend()
        plt.tight_layout()
        plt.savefig(f'statistic_figure/base_{primary_name}_mix_two.pdf')
        plt.show()

plot_grouped_bars_with_age_gaps(rafdb_data, 'gender', ['race', 'age'], gender_map, {'race': race_map, 'age': age_map}, custom_sort_gender)
plot_grouped_bars_with_age_gaps(rafdb_data, 'race', ['gender', 'age'], race_map, {'gender': gender_map, 'age': age_map}, custom_sort_race)
plot_grouped_bars_with_age_gaps(rafdb_data, 'age', ['gender', 'race'], age_map, {'gender': gender_map, 'race': race_map}, custom_sort_age)



'''
radar plots for different attributes
'''
# absolute numbers
def plot_radar_for_category(data, category, subgroup_map, colors, line_styles, emotion_map):
    common_labels = list(emotion_map.values())  # Use emotion names from the map
    common_num_vars = len(common_labels)
    fig, ax = plt.subplots(figsize=fig_size, subplot_kw=dict(polar=True))
    common_angles = np.linspace(0, 2 * np.pi, common_num_vars, endpoint=False).tolist()
    common_angles += common_angles[:1]  # Ensure circular data

    # Plot data for the given category and subgroup
    for key, label in subgroup_map.items():
        subgroup_data = data[data[category] == key]['Emotion'].map(emotion_map).value_counts()
        subgroup_stats = [subgroup_data.get(l, 0) for l in common_labels]
        subgroup_stats += subgroup_stats[:1]  # Close the radar chart

        # Plot
        ax.plot(common_angles, subgroup_stats, color=colors[label], linestyle=line_styles[label], linewidth=2, label=f"{label}")
        ax.fill(common_angles, subgroup_stats, color=colors[label], alpha=0.25)

    # Set labels and legend
    ax.set_xticks(common_angles[:-1])
    ax.set_xticklabels(common_labels)
    ax.set_yticklabels([])  # Hide y-axis labels
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.15))

    plt.title(f'Emotion by {category}', fontdict=title_font)
    plt.savefig(f'statistic_figure/radar_plot_{category}.pdf')
    plt.show()

# Colors and line styles for each subgroup
colors = {
    'Male': 'blue',
    'Female': 'red',
    'Caucasian': 'green',
    'African-American': 'brown',
    'Asian': 'purple',
    '0-3': 'magenta',
    '4-19': 'orange',
    '20-39': 'yellow',
    '40-69': 'gray',
    '70+': 'black'
}
line_styles = {
    'Male': '-',
    'Female': '--',
    'Caucasian': '-',
    'African-American': '--',
    'Asian': '-.',
    '0-3': '-',
    '4-19': '--',
    '20-39': '-.',
    '40-69': ':',
    '70+': 'dotted'
}

# Generate radar charts for each category
plot_radar_for_category(rafdb_data, 'gender', gender_map, colors, line_styles, emotion_map)
plot_radar_for_category(rafdb_data, 'race', race_map, colors, line_styles, emotion_map)
plot_radar_for_category(rafdb_data, 'age', age_map, colors, line_styles, emotion_map)


# frequency of radar plots
def plot_radar_for_frequency(data, category, subgroup_map, colors, line_styles, emotion_map):
    common_labels = list(emotion_map.values())  # Use emotion names from the map
    common_num_vars = len(common_labels)
    fig, ax = plt.subplots(figsize=fig_size, subplot_kw=dict(polar=True))
    common_angles = np.linspace(0, 2 * np.pi, common_num_vars, endpoint=False).tolist()
    common_angles += common_angles[:1]  # Ensure circular data

    # Plot normalized data for the given category and subgroup
    for key, label in subgroup_map.items():
        category_data = data[data[category] == key]
        total = category_data.shape[0]
        subgroup_data = category_data['Emotion'].map(emotion_map).value_counts(normalize=True) * 100
        subgroup_stats = [subgroup_data.get(l, 0) for l in common_labels]
        subgroup_stats += subgroup_stats[:1]  # Close the radar chart

        # Plot
        ax.plot(common_angles, subgroup_stats, color=colors[label], linestyle=line_styles[label], linewidth=2, label=f"{label}")
        ax.fill(common_angles, subgroup_stats, color=colors[label], alpha=0.25)

    # Set labels and legend
    ax.set_xticks(common_angles[:-1])
    ax.set_xticklabels(common_labels)
    ax.set_yticklabels([])  # Hide y-axis labels
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.15))

    plt.title(f'Emotion frequency by {category}', fontdict=title_font)
    plt.savefig(f'statistic_figure/radar_plot_frequency_{category}.pdf')
    plt.show()


plot_radar_for_frequency(rafdb_data, 'gender', gender_map, colors, line_styles, emotion_map)
plot_radar_for_frequency(rafdb_data, 'race', race_map, colors, line_styles, emotion_map)
plot_radar_for_frequency(rafdb_data, 'age', age_map, colors, line_styles, emotion_map)