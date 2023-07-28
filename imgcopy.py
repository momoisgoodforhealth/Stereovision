import shutil

total_photos = 66
photo_counter = 0
print ('Main cycle start')

dst_counter=122
while photo_counter != total_photos:
    srcleftName = './underL/'+str(photo_counter)+'.png'
    srcrightName = './underR/'+str(photo_counter)+'.png'
    dstleftName = './cleanL/'+str(dst_counter)+'.png'
    dstrightName = './cleanR/'+str(dst_counter)+'.png'
    try:
        shutil.copy(srcleftName, dstleftName)
        shutil.copy(srcrightName, dstrightName)
        print("File copied successfully."+str(dst_counter))
        photo_counter=photo_counter+1
        dst_counter=dst_counter+1
 
# If source and destination are same
    except shutil.SameFileError:
        print("Source and destination represents the same file.")
 
# If there is any permission issue
    except PermissionError:
        print("Permission denied.")
 
# For other errors
    except:
        print("Error occurred while copying file.")