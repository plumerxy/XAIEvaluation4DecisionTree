def findDecision(obj): #obj[0]: mean radius, obj[1]: mean texture, obj[2]: mean perimeter, obj[3]: mean area, obj[4]: mean smoothness, obj[5]: mean compactness, obj[6]: mean concavity, obj[7]: mean concave points, obj[8]: mean symmetry, obj[9]: mean fractal dimension, obj[10]: radius error, obj[11]: texture error, obj[12]: perimeter error, obj[13]: area error, obj[14]: smoothness error, obj[15]: compactness error, obj[16]: concavity error, obj[17]: concave points error, obj[18]: symmetry error, obj[19]: fractal dimension error, obj[20]: worst radius, obj[21]: worst texture, obj[22]: worst perimeter, obj[23]: worst area, obj[24]: worst smoothness, obj[25]: worst compactness, obj[26]: worst concavity, obj[27]: worst concave points, obj[28]: worst symmetry, obj[29]: worst fractal dimension
   # {"feature": "worst area", "instances": 398, "metric_value": 33.7746, "depth": 1}
   if obj[23]<=867.9324120603014:
      # {"feature": "mean radius", "instances": 266, "metric_value": 27.2695, "depth": 2}
      if obj[0]>12.110887218045113:
         # {"feature": "worst radius", "instances": 137, "metric_value": 17.4137, "depth": 3}
         if obj[20]>13.734736385929422:
            # {"feature": "worst concave points", "instances": 108, "metric_value": 14.1412, "depth": 4}
            if obj[27]<=0.1424247100247874:
               # {"feature": "fractal dimension error", "instances": 93, "metric_value": 16.4546, "depth": 5}
               if obj[19]<=0.00327037741935484:
                  # {"feature": "worst smoothness", "instances": 64, "metric_value": 12.5202, "depth": 6}
                  if obj[24]>0.1165796875:
                     # {"feature": "mean compactness", "instances": 33, "metric_value": 8.35, "depth": 7}
                     if obj[5]<=0.12287378358556056:
                        # {"feature": "concave points error", "instances": 31, "metric_value": 9.0905, "depth": 8}
                        if obj[17]<=0.014690834323316106:
                           # {"feature": "mean area", "instances": 29, "metric_value": 10.0488, "depth": 9}
                           if obj[3]<=559.8310344827585:
                              # {"feature": "mean texture", "instances": 16, "metric_value": 7.0418, "depth": 10}
                              if obj[1]>17.02:
                                 # {"feature": "texture error", "instances": 10, "metric_value": 5.6569, "depth": 11}
                                 if obj[11]<=1.373:
                                    return '1'
                                 elif obj[11]>1.373:
                                    return '0'
                                 else:
                                    return '0'
                              elif obj[1]<=17.02:
                                 return '1'
                              else:
                                 return '1'
                           elif obj[3]>559.8310344827585:
                              return '1'
                           else:
                              return '1'
                        elif obj[17]>0.014690834323316106:
                           return '0'
                        else:
                           return '0'
                     elif obj[5]>0.12287378358556056:
                        return '0'
                     else:
                        return '0'
                  elif obj[24]<=0.1165796875:
                     # {"feature": "mean fractal dimension", "instances": 31, "metric_value": 9.7657, "depth": 7}
                     if obj[9]<=0.05634387096774193:
                        # {"feature": "mean texture", "instances": 18, "metric_value": 6.7301, "depth": 8}
                        if obj[1]>16.82:
                           # {"feature": "mean perimeter", "instances": 12, "metric_value": 5.2518, "depth": 9}
                           if obj[2]<=95.5:
                              # {"feature": "worst symmetry", "instances": 11, "metric_value": 5.8863, "depth": 10}
                              if obj[28]<=0.2663:
                                 return '1'
                              elif obj[28]>0.2663:
                                 return '0'
                              else:
                                 return '0'
                           elif obj[2]>95.5:
                              return '0'
                           else:
                              return '0'
                        elif obj[1]<=16.82:
                           return '1'
                        else:
                           return '1'
                     elif obj[9]>0.05634387096774193:
                        return '1'
                     else:
                        return '1'
                  else:
                     return '1'
               elif obj[19]>0.00327037741935484:
                  return '1'
               else:
                  return '1'
            elif obj[27]>0.1424247100247874:
               # {"feature": "symmetry error", "instances": 15, "metric_value": 7.5188, "depth": 5}
               if obj[18]>0.01454:
                  return '0'
               elif obj[18]<=0.01454:
                  return '1'
               else:
                  return '1'
            else:
               return '0'
         elif obj[20]<=13.734736385929422:
            return '1'
         else:
            return '1'
      elif obj[0]<=12.110887218045113:
         # {"feature": "worst radius", "instances": 129, "metric_value": 21.3437, "depth": 3}
         if obj[20]>11.853209302325581:
            # {"feature": "worst perimeter", "instances": 72, "metric_value": 15.1287, "depth": 4}
            if obj[22]>83.07625:
               # {"feature": "mean symmetry", "instances": 38, "metric_value": 9.7739, "depth": 5}
               if obj[8]>0.17988157894736845:
                  # {"feature": "mean area", "instances": 22, "metric_value": 7.6921, "depth": 6}
                  if obj[3]>371.4609378875613:
                     # {"feature": "mean texture", "instances": 20, "metric_value": 8.0825, "depth": 7}
                     if obj[1]<=18.33:
                        # {"feature": "mean perimeter", "instances": 12, "metric_value": 6.1046, "depth": 8}
                        if obj[2]<=78.11:
                           return '1'
                        elif obj[2]>78.11:
                           return '0'
                        else:
                           return '0'
                     elif obj[1]>18.33:
                        return '1'
                     else:
                        return '1'
                  elif obj[3]<=371.4609378875613:
                     return '0'
                  else:
                     return '0'
               elif obj[8]<=0.17988157894736845:
                  # {"feature": "mean texture", "instances": 16, "metric_value": 7.0418, "depth": 6}
                  if obj[1]>17.07:
                     # {"feature": "radius error", "instances": 10, "metric_value": 5.6569, "depth": 7}
                     if obj[10]<=0.4384:
                        return '1'
                     elif obj[10]>0.4384:
                        return '0'
                     else:
                        return '0'
                  elif obj[1]<=17.07:
                     return '1'
                  else:
                     return '1'
               else:
                  return '1'
            elif obj[22]<=83.07625:
               return '1'
            else:
               return '1'
         elif obj[20]<=11.853209302325581:
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
