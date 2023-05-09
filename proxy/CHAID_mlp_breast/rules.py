def findDecision(obj): #obj[0]: mean radius, obj[1]: mean texture, obj[2]: mean perimeter, obj[3]: mean area, obj[4]: mean smoothness, obj[5]: mean compactness, obj[6]: mean concavity, obj[7]: mean concave points, obj[8]: mean symmetry, obj[9]: mean fractal dimension, obj[10]: radius error, obj[11]: texture error, obj[12]: perimeter error, obj[13]: area error, obj[14]: smoothness error, obj[15]: compactness error, obj[16]: concavity error, obj[17]: concave points error, obj[18]: symmetry error, obj[19]: fractal dimension error, obj[20]: worst radius, obj[21]: worst texture, obj[22]: worst perimeter, obj[23]: worst area, obj[24]: worst smoothness, obj[25]: worst compactness, obj[26]: worst concavity, obj[27]: worst concave points, obj[28]: worst symmetry, obj[29]: worst fractal dimension
   # {"feature": "worst area", "instances": 398, "metric_value": 34.1215, "depth": 1}
   if obj[23]<=867.9324120603014:
      # {"feature": "worst smoothness", "instances": 266, "metric_value": 27.7602, "depth": 2}
      if obj[24]>0.1269656015037594:
         # {"feature": "smoothness error", "instances": 137, "metric_value": 17.7167, "depth": 3}
         if obj[14]<=0.008431116788321168:
            # {"feature": "worst concave points", "instances": 80, "metric_value": 14.0576, "depth": 4}
            if obj[27]<=0.15056006884564277:
               # {"feature": "mean texture", "instances": 70, "metric_value": 14.839, "depth": 5}
               if obj[1]>17.039714285714282:
                  # {"feature": "fractal dimension error", "instances": 37, "metric_value": 10.2952, "depth": 6}
                  if obj[19]<=0.003135891891891892:
                     # {"feature": "mean concavity", "instances": 24, "metric_value": 8.0302, "depth": 7}
                     if obj[6]<=0.09341892689748924:
                        # {"feature": "mean symmetry", "instances": 22, "metric_value": 8.5572, "depth": 8}
                        if obj[8]<=0.16851818181818184:
                           # {"feature": "mean radius", "instances": 13, "metric_value": 6.3132, "depth": 9}
                           if obj[0]<=14.03:
                              return '1'
                           elif obj[0]>14.03:
                              return '0'
                           else:
                              return '0'
                        elif obj[8]>0.16851818181818184:
                           return '1'
                        else:
                           return '1'
                     elif obj[6]>0.09341892689748924:
                        return '0'
                     else:
                        return '0'
                  elif obj[19]>0.003135891891891892:
                     return '1'
                  else:
                     return '1'
               elif obj[1]<=17.039714285714282:
                  # {"feature": "mean symmetry", "instances": 33, "metric_value": 10.8106, "depth": 6}
                  if obj[8]<=0.17462727272727271:
                     # {"feature": "mean radius", "instances": 18, "metric_value": 7.5793, "depth": 7}
                     if obj[0]>11.71:
                        # {"feature": "mean compactness", "instances": 11, "metric_value": 5.8863, "depth": 8}
                        if obj[5]<=0.09242:
                           return '1'
                        elif obj[5]>0.09242:
                           return '0'
                        else:
                           return '0'
                     elif obj[0]<=11.71:
                        return '1'
                     else:
                        return '1'
                  elif obj[8]>0.17462727272727271:
                     return '1'
                  else:
                     return '1'
               else:
                  return '1'
            elif obj[27]>0.15056006884564277:
               # {"feature": "mean texture", "instances": 10, "metric_value": 5.6569, "depth": 5}
               if obj[1]>15.18:
                  return '0'
               elif obj[1]<=15.18:
                  return '1'
               else:
                  return '1'
            else:
               return '0'
         elif obj[14]>0.008431116788321168:
            # {"feature": "mean radius", "instances": 57, "metric_value": 13.0288, "depth": 4}
            if obj[0]>10.951543859649124:
               # {"feature": "area error", "instances": 30, "metric_value": 8.9684, "depth": 5}
               if obj[13]<=33.37887815345745:
                  return '1'
               elif obj[13]>33.37887815345745:
                  # {"feature": "mean texture", "instances": 5, "metric_value": 4.2426, "depth": 6}
                  if obj[1]>13.17:
                     return '0'
                  elif obj[1]<=13.17:
                     return '1'
                  else:
                     return '1'
               else:
                  return '0'
            elif obj[0]<=10.951543859649124:
               return '1'
            else:
               return '1'
         else:
            return '1'
      elif obj[24]<=0.1269656015037594:
         # {"feature": "worst symmetry", "instances": 129, "metric_value": 21.6803, "depth": 3}
         if obj[28]<=0.2574953488372093:
            # {"feature": "worst fractal dimension", "instances": 69, "metric_value": 15.2043, "depth": 4}
            if obj[29]<=0.07089710144927537:
               # {"feature": "mean perimeter", "instances": 42, "metric_value": 11.1193, "depth": 5}
               if obj[2]>81.55523809523808:
                  # {"feature": "smoothness error", "instances": 22, "metric_value": 7.6752, "depth": 6}
                  if obj[14]>0.0048878181818181815:
                     # {"feature": "concave points error", "instances": 11, "metric_value": 6.2426, "depth": 7}
                     if obj[17]<=0.01183:
                        return '1'
                     elif obj[17]>0.01183:
                        return '0'
                     else:
                        return '0'
                  elif obj[14]<=0.0048878181818181815:
                     return '1'
                  else:
                     return '1'
               elif obj[2]<=81.55523809523808:
                  # {"feature": "mean radius", "instances": 20, "metric_value": 8.0825, "depth": 6}
                  if obj[0]>11.13:
                     # {"feature": "mean smoothness", "instances": 12, "metric_value": 6.1046, "depth": 7}
                     if obj[4]<=0.08752:
                        return '1'
                     elif obj[4]>0.08752:
                        return '0'
                     else:
                        return '0'
                  elif obj[0]<=11.13:
                     return '1'
                  else:
                     return '1'
               else:
                  return '1'
            elif obj[29]>0.07089710144927537:
               return '1'
            else:
               return '1'
         elif obj[28]>0.2574953488372093:
            return '1'
         else:
            return '1'
      else:
         return '1'
   elif obj[23]>867.9324120603014:
      # {"feature": "concavity error", "instances": 132, "metric_value": 20.6407, "depth": 2}
      if obj[16]<=0.03929462121212121:
         # {"feature": "area error", "instances": 76, "metric_value": 14.4232, "depth": 3}
         if obj[13]<=58.907500000000006:
            # {"feature": "mean texture", "instances": 47, "metric_value": 10.4813, "depth": 4}
            if obj[1]>15.840079376485372:
               # {"feature": "mean fractal dimension", "instances": 41, "metric_value": 11.6045, "depth": 5}
               if obj[9]<=0.05937878048780488:
                  # {"feature": "mean compactness", "instances": 24, "metric_value": 8.2593, "depth": 6}
                  if obj[5]<=0.09101083333333333:
                     # {"feature": "compactness error", "instances": 15, "metric_value": 7.099, "depth": 7}
                     if obj[15]<=0.0163:
                        return '0'
                     elif obj[15]>0.0163:
                        return '1'
                     else:
                        return '1'
                  elif obj[5]>0.09101083333333333:
                     return '0'
                  else:
                     return '0'
               elif obj[9]>0.05937878048780488:
                  return '0'
               else:
                  return '0'
            elif obj[1]<=15.840079376485372:
               # {"feature": "mean radius", "instances": 6, "metric_value": 4.5765, "depth": 5}
               if obj[0]<=17.85:
                  return '1'
               elif obj[0]>17.85:
                  return '0'
               else:
                  return '0'
            else:
               return '1'
         elif obj[13]>58.907500000000006:
            return '0'
         else:
            return '0'
      elif obj[16]>0.03929462121212121:
         return '0'
      else:
         return '0'
   else:
      return '0'
