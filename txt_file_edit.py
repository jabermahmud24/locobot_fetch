# Open the input file
# with open('arm_velocity_iteration.txt', 'r') as file:
#     # Read the lines
#     lines = file.readlines()

# # Open the output file
# with open('hard_code_arm_vel.txt', 'w') as file:
#     # Iterate over the lines
#     for line in lines:
#         # Add '[' at the beginning and ']' at the end of each line
#         modified_line = '[' + line.strip() + '],\n'
#         # Write the modified line to the output file
#         file.write(modified_line)


# Open the input file
with open('hard_code_arm_waypoint_without_comma.txt', 'r') as file:
    # Read the lines
    lines = file.readlines()

# Open the output file
with open('hard_code_arm_waypoint.txt', 'w') as file:
    # Iterate over the lines
    for line in lines:
        # Split the line into individual entries
        entries = line.strip().split()
        # Join the entries with commas and add '[' at the beginning and ']' at the end
        modified_line = '[' + ', '.join(entries) + '],\n'
        # Write the modified line to the output file
        file.write(modified_line)


# input_file = "arm_velocity_iteration.txt"
# output_file = "hard_code_arm_waypoint_without_comma.txt"

# # Read the input file
# with open(input_file, "r") as file:
#     lines = file.readlines()

# # Process each line
# modified_lines = []
# for line in lines:
#     entries = line.strip().split()  # Split by spaces
#     modified_entries = [str(float(entry) * 0.1) for entry in entries]
#     modified_lines.append(" ".join(modified_entries) + "\n")  # Join with spaces

# # Write the modified lines to the output file
# with open(output_file, "w") as file:
#     file.writelines(modified_lines)

# print("Output has been written to", output_file)
