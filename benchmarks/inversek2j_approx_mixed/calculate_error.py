import sys
import numpy as np
def parse_file(file_t):
   result = []
   with open(file_t) as res_file:
        line_num = 0
        for line in res_file:
            #print line
            line_num = line_num+1
            array = line.split(' ')
            #print (array[2:])
            for target in array[2:]:
                if len(target) > 0 and ord(target[0]) != 10:
                    try:
                        result.append(float(target))
                    except:
                        print ("cant convert " + target )
                        print (ord(target[0]))
   return result
def calculate_err (array_1, array_2):
    #calculate avg rel err
    err = 0.0;
    err_array = [];
    rmse = 0.0;
    sum_abs = 0.0;
    for i in range(len(array_1)):
        #if i == 0:
        #    print "%f %f %f "%(array_1[i],array_2[i], abs((array_1[i]- array_2[i])/ array_2[i]))
        temp_err = 0
        if array_2[i]!=0:
            temp_err = abs((array_1[i]- array_2[i])/ array_2[i])
        else:
            temp_err = abs(array_1[i])
        sum_abs += abs((array_1[i]- array_2[i]))
        if (temp_err > 1):
			temp_err = 1

        err += temp_err
        rmse = rmse + (array_1[i]- array_2[i])*(array_1[i]- array_2[i]);
        err_array.append(temp_err)

    #print ("sum abs %f "%(sum_abs))
    return err/float(len(array_1)), err_array
def main(argv):
    if (len(argv) != 2):
        print ("usage: python calculate_error.py file1.txt file2.txt")
    file1 = argv[0]
    file2 = argv[1]
    array_1 = parse_file(file1)
    array_2 = parse_file(file2)
    if (len(array_1)!= len(array_2)):
        print ("len mismatch by ",(len(array_1) - len(array_2)))
    print ("len array " ,(len(array_1)))
    #print (array_1[:10])
    #print (array_2[:10])
    err,err_array = calculate_err (array_1, array_2)
    bins = np.arange(0,1.0,0.1)
    hist = np.histogram(err_array,bins)
    print ("cdf : ")
    pdf = hist[0]/float(len(err_array))
    cdf = np.cumsum(pdf)
    print (list(cdf))
    print ("rel err:  ",(err))
if __name__ == '__main__':
    main(sys.argv[1:])
