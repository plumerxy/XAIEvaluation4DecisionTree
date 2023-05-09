def findDecision(obj): #obj[0]: alcohol, obj[1]: malic acid, obj[2]: ash, obj[3]: alcalinityofash, obj[4]: magnesium, obj[5]: total phenols, obj[6]: flavanoids, obj[7]: nonflavanoid phenols, obj[8]: proanthocyanins, obj[9]: color intensity, obj[10]: hue, obj[11]: od280, obj[12]: proline
   # {"feature": "proline", "instances": 124, "metric_value": 22.3017, "depth": 1}
   if obj[12]<=767.3306451612904:
      # {"feature": "color intensity", "instances": 75, "metric_value": 23.5622, "depth": 2}
      if obj[9]<=4.75533332:
         # {"feature": "hue", "instances": 47, "metric_value": 18.915, "depth": 3}
         if obj[10]<=1.059787234042553:
            # {"feature": "od280", "instances": 25, "metric_value": 12.1745, "depth": 4}
            if obj[11]>1.4381838441605213:
               # {"feature": "flavanoids", "instances": 23, "metric_value": 13.0931, "depth": 5}
               if obj[6]>0.52:
                  # {"feature": "alcohol", "instances": 21, "metric_value": 8.3251, "depth": 6}
                  if obj[0]>12.25:
                     # {"feature": "ash", "instances": 12, "metric_value": 6.1046, "depth": 7}
                     if obj[2]<=2.73:
                        return '1'
                     elif obj[2]>2.73:
                        return '0'
                     else:
                        return '0'
                  elif obj[0]<=12.25:
                     return '1'
                  else:
                     return '1'
               elif obj[6]<=0.52:
                  return '2'
               else:
                  return '2'
            elif obj[11]<=1.4381838441605213:
               return '2'
            else:
               return '2'
         elif obj[10]>1.059787234042553:
            return '1'
         else:
            return '1'
      elif obj[9]>4.75533332:
         # {"feature": "ash", "instances": 28, "metric_value": 8.6564, "depth": 3}
         if obj[2]>2.02:
            # {"feature": "alcohol", "instances": 26, "metric_value": 9.4373, "depth": 4}
            if obj[0]<=13.071538461538463:
               # {"feature": "malic acid", "instances": 15, "metric_value": 6.7639, "depth": 5}
               if obj[1]>2.67:
                  # {"feature": "magnesium", "instances": 9, "metric_value": 5.4142, "depth": 6}
                  if obj[4]>85:
                     return '2'
                  elif obj[4]<=85:
                     return '1'
                  else:
                     return '1'
               elif obj[1]<=2.67:
                  return '2'
               else:
                  return '2'
            elif obj[0]>13.071538461538463:
               return '2'
            else:
               return '2'
         elif obj[2]<=2.02:
            return '1'
         else:
            return '1'
      else:
         return '2'
   elif obj[12]>767.3306451612904:
      # {"feature": "ash", "instances": 49, "metric_value": 18.0935, "depth": 2}
      if obj[2]<=2.4171428571428573:
         # {"feature": "malic acid", "instances": 27, "metric_value": 13.1924, "depth": 3}
         if obj[1]<=2.456774866598152:
            # {"feature": "alcohol", "instances": 24, "metric_value": 8.9302, "depth": 4}
            if obj[0]>12.95263572832166:
               return '0'
            elif obj[0]<=12.95263572832166:
               return '1'
            else:
               return '1'
         elif obj[1]>2.456774866598152:
            return '2'
         else:
            return '2'
      elif obj[2]>2.4171428571428573:
         # {"feature": "alcalinityofash", "instances": 22, "metric_value": 8.5572, "depth": 3}
         if obj[3]>16.8:
            # {"feature": "magnesium", "instances": 13, "metric_value": 6.3132, "depth": 4}
            if obj[4]<=121:
               return '0'
            elif obj[4]>121:
               return '1'
            else:
               return '1'
         elif obj[3]<=16.8:
            return '0'
         else:
            return '0'
      else:
         return '0'
   else:
      return '0'
