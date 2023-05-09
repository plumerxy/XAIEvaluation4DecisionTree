def findDecision(obj): #obj[0]: alcohol, obj[1]: malic acid, obj[2]: ash, obj[3]: alcalinityofash, obj[4]: magnesium, obj[5]: total phenols, obj[6]: flavanoids, obj[7]: nonflavanoid phenols, obj[8]: proanthocyanins, obj[9]: color intensity, obj[10]: hue, obj[11]: od280, obj[12]: proline
   # {"feature": "proline", "instances": 124, "metric_value": 1.5635, "depth": 1}
   if obj[12]<=767.3306451612904:
      # {"feature": "color intensity", "instances": 75, "metric_value": 1.0553, "depth": 2}
      if obj[9]<=4.75533332:
         # {"feature": "flavanoids", "instances": 47, "metric_value": 0.5657, "depth": 3}
         if obj[6]>0.6100951433481145:
            # {"feature": "od280", "instances": 44, "metric_value": 0.3122, "depth": 4}
            if obj[11]>1.29:
               # {"feature": "alcohol", "instances": 43, "metric_value": 0.1594, "depth": 5}
               if obj[0]<=12.819601812606388:
                  return '1'
               elif obj[0]>12.819601812606388:
                  # {"feature": "malic acid", "instances": 4, "metric_value": 0.8113, "depth": 6}
                  if obj[1]<=1.66:
                     return '1'
                  elif obj[1]>1.66:
                     return '0'
                  else:
                     return '0'
               else:
                  return '1'
            elif obj[11]<=1.29:
               return '2'
            else:
               return '2'
         elif obj[6]<=0.6100951433481145:
            return '2'
         else:
            return '2'
      elif obj[9]>4.75533332:
         # {"feature": "ash", "instances": 28, "metric_value": 0.4912, "depth": 3}
         if obj[2]>2.02:
            # {"feature": "magnesium", "instances": 26, "metric_value": 0.2352, "depth": 4}
            if obj[4]>85:
               return '2'
            elif obj[4]<=85:
               # {"feature": "alcohol", "instances": 2, "metric_value": 1.0, "depth": 5}
               if obj[0]>12.51:
                  return '1'
               elif obj[0]<=12.51:
                  return '2'
               else:
                  return '2'
            else:
               return '1'
         elif obj[2]<=2.02:
            return '1'
         else:
            return '1'
      else:
         return '2'
   elif obj[12]>767.3306451612904:
      # {"feature": "hue", "instances": 49, "metric_value": 0.7324, "depth": 2}
      if obj[10]>0.7277812735439918:
         # {"feature": "alcohol", "instances": 46, "metric_value": 0.4262, "depth": 3}
         if obj[0]>12.50630219688631:
            # {"feature": "alcalinityofash", "instances": 43, "metric_value": 0.1594, "depth": 4}
            if obj[3]<=23.008484384990957:
               return '0'
            elif obj[3]>23.008484384990957:
               return '1'
            else:
               return '1'
         elif obj[0]<=12.50630219688631:
            return '1'
         else:
            return '1'
      elif obj[10]<=0.7277812735439918:
         return '2'
      else:
         return '2'
   else:
      return '0'
