def findDecision(obj): #obj[0]: mean radius, obj[1]: mean texture, obj[2]: mean perimeter, obj[3]: mean area, obj[4]: mean smoothness, obj[5]: mean compactness, obj[6]: mean concavity, obj[7]: mean concave points, obj[8]: mean symmetry, obj[9]: mean fractal dimension, obj[10]: radius error, obj[11]: texture error, obj[12]: perimeter error, obj[13]: area error, obj[14]: smoothness error, obj[15]: compactness error, obj[16]: concavity error, obj[17]: concave points error, obj[18]: symmetry error, obj[19]: fractal dimension error, obj[20]: worst radius, obj[21]: worst texture, obj[22]: worst perimeter, obj[23]: worst area, obj[24]: worst smoothness, obj[25]: worst compactness, obj[26]: worst concavity, obj[27]: worst concave points, obj[28]: worst symmetry, obj[29]: worst fractal dimension
   # {"feature": "worst area", "instances": 398, "metric_value": 0.9462, "depth": 1}
   if obj[23]<=867.9324120603014:
      # {"feature": "worst concave points", "instances": 266, "metric_value": 0.385, "depth": 2}
      if obj[27]<=0.1662060496374062:
         # {"feature": "symmetry error", "instances": 257, "metric_value": 0.2722, "depth": 3}
         if obj[18]>0.007882:
            # {"feature": "mean perimeter", "instances": 256, "metric_value": 0.2557, "depth": 4}
            if obj[2]<=99.83852971450288:
               # {"feature": "worst smoothness", "instances": 254, "metric_value": 0.2394, "depth": 5}
               if obj[24]<=0.1888153771727885:
                  # {"feature": "mean concave points", "instances": 252, "metric_value": 0.2223, "depth": 6}
                  if obj[7]<=0.04077245089778336:
                     # {"feature": "radius error", "instances": 213, "metric_value": 0.0431, "depth": 7}
                     if obj[10]<=0.5905597140316046:
                        return '1'
                     elif obj[10]>0.5905597140316046:
                        # {"feature": "mean texture", "instances": 4, "metric_value": 0.8113, "depth": 8}
                        if obj[1]>18.14:
                           return '1'
                        elif obj[1]<=18.14:
                           return '0'
                        else:
                           return '0'
                     else:
                        return '1'
                  elif obj[7]>0.04077245089778336:
                     # {"feature": "mean smoothness", "instances": 39, "metric_value": 0.7321, "depth": 7}
                     if obj[4]>0.08682:
                        # {"feature": "mean fractal dimension", "instances": 38, "metric_value": 0.6892, "depth": 8}
                        if obj[9]>0.05636:
                           # {"feature": "worst texture", "instances": 37, "metric_value": 0.6395, "depth": 9}
                           if obj[21]<=34.779998318112774:
                              # {"feature": "worst concavity", "instances": 36, "metric_value": 0.5813, "depth": 10}
                              if obj[26]<=0.27358:
                                 return '1'
                              elif obj[26]>0.27358:
                                 # {"feature": "worst radius", "instances": 14, "metric_value": 0.9403, "depth": 11}
                                 if obj[20]<=15.85:
                                    # {"feature": "mean radius", "instances": 10, "metric_value": 0.469, "depth": 12}
                                    if obj[0]<=13.66:
                                       return '1'
                                    elif obj[0]>13.66:
                                       # {"feature": "mean texture", "instances": 2, "metric_value": 1.0, "depth": 13}
                                       if obj[1]<=15.18:
                                          return '1'
                                       elif obj[1]>15.18:
                                          return '0'
                                       else:
                                          return '0'
                                    else:
                                       return '1'
                                 elif obj[20]>15.85:
                                    return '0'
                                 else:
                                    return '0'
                              else:
                                 return '1'
                           elif obj[21]>34.779998318112774:
                              return '0'
                           else:
                              return '0'
                        elif obj[9]<=0.05636:
                           return '0'
                        else:
                           return '0'
                     elif obj[4]<=0.08682:
                        return '0'
                     else:
                        return '0'
                  else:
                     return '1'
               elif obj[24]>0.1888153771727885:
                  # {"feature": "mean radius", "instances": 2, "metric_value": 1.0, "depth": 6}
                  if obj[0]>9.268:
                     return '0'
                  elif obj[0]<=9.268:
                     return '1'
                  else:
                     return '1'
               else:
                  return '0'
            elif obj[2]>99.83852971450288:
               # {"feature": "mean radius", "instances": 2, "metric_value": 1.0, "depth": 5}
               if obj[0]<=15.37:
                  return '0'
               elif obj[0]>15.37:
                  return '1'
               else:
                  return '1'
            else:
               return '0'
         elif obj[18]<=0.007882:
            return '0'
         else:
            return '0'
      elif obj[27]>0.1662060496374062:
         # {"feature": "mean radius", "instances": 9, "metric_value": 0.5033, "depth": 3}
         if obj[0]>9.029:
            return '0'
         elif obj[0]<=9.029:
            return '1'
         else:
            return '1'
      else:
         return '0'
   elif obj[23]>867.9324120603014:
      # {"feature": "mean texture", "instances": 132, "metric_value": 0.2991, "depth": 2}
      if obj[1]>13.52157226931291:
         # {"feature": "worst smoothness", "instances": 129, "metric_value": 0.1994, "depth": 3}
         if obj[24]>0.08774:
            # {"feature": "worst texture", "instances": 128, "metric_value": 0.1603, "depth": 4}
            if obj[21]>18.133739654174455:
               # {"feature": "mean concavity", "instances": 126, "metric_value": 0.1176, "depth": 5}
               if obj[6]>0.08792098879984327:
                  return '0'
               elif obj[6]<=0.08792098879984327:
                  # {"feature": "mean symmetry", "instances": 17, "metric_value": 0.5226, "depth": 6}
                  if obj[8]>0.1495:
                     # {"feature": "mean smoothness", "instances": 16, "metric_value": 0.3373, "depth": 7}
                     if obj[4]<=0.09215:
                        return '0'
                     elif obj[4]>0.09215:
                        # {"feature": "mean radius", "instances": 2, "metric_value": 1.0, "depth": 8}
                        if obj[0]<=14.68:
                           return '0'
                        elif obj[0]>14.68:
                           return '1'
                        else:
                           return '1'
                     else:
                        return '0'
                  elif obj[8]<=0.1495:
                     return '1'
                  else:
                     return '1'
               else:
                  return '0'
            elif obj[21]<=18.133739654174455:
               # {"feature": "mean radius", "instances": 2, "metric_value": 1.0, "depth": 5}
               if obj[0]>14.76:
                  return '0'
               elif obj[0]<=14.76:
                  return '1'
               else:
                  return '1'
            else:
               return '0'
         elif obj[24]<=0.08774:
            return '1'
         else:
            return '1'
      elif obj[1]<=13.52157226931291:
         return '1'
      else:
         return '1'
   else:
      return '0'
