using CSV
using DataFrames

# create an empty DataFrame
df = DataFrame()

# loop over each file and append its contents to the DataFrame
for i in 1:16
    filename = "train_dataset_riya_$i.txt"
    temp_df = CSV.read(filename, header=true, delim='\t', DataFrame)
    global df = vcat(df, temp_df)
end

# write the concatenated DataFrame to a new file
CSV.write("concatenated_train_files.txt", df)