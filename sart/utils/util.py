
# Save the results of the robust radius verification
def save_radius_result(i, delta, cost_time, file_url):
        file_url.write(str(i) + " -- delta_base : " + str(delta) + "\n")
        file_url.write(str(i) + " -- time : " + str(cost_time) + "\n")


def save_number_result(i, flag, cost_time, file_url):
        file_url.write(str(i) + " -- is__verify : " + str(flag) + "\n")
        file_url.write(str(i) + " -- time : " + str(cost_time) + "\n")