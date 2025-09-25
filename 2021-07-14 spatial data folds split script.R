# Prepare Fairbanks CWPP ground ice data for training models on the 
# ground ice extent region.
# 15-July-2021


library(tidyverse)

# I don't want to see scientific notation
options(scipen=8) 

# Set random seed for reproducibility
set.seed(42)

#############################################################
# My functions
#############################################################

# # FUNCTION: rectangular_partitions
# # RETURNS:  a vector indicating which observation belongs 
# #           to test data (1), and which to training data (0)
# # PARAMETERS:
# #   XCoord, YCoord: x and y coordinates of the observations
# #   num_rectangles: desired number of rectangles
# #   test_pct: desired proportion of observations in the test data
# rectangular_partitions <- function(
#   XCoord, YCoord, 
#   num_rectangles=1000, 
#   test_pct=0.25){
#   
#   # Put the coordinates in a local data frame
#   df <- data.frame(XCoord, YCoord)
#   
#   # resulting side widths of each rectangle
#   X_width <- (max(XCoord)-min(XCoord))/ceiling(sqrt(num_rectangles)) 
#   Y_width <- (max(YCoord)-min(YCoord))/ceiling(sqrt(num_rectangles)) 
#   
#   
#   # X and Y coordinate markers that separate the rectangles
#   X_sep <- seq(min(XCoord), 
#                max(XCoord)-X_width, by=X_width)
#   Y_sep <- seq(min(YCoord), 
#                max(YCoord)-Y_width, by=Y_width)
#   
#   XY_sep <- crossing(X_sep, Y_sep)
#   
#   # randomly select which of the rectangles will be in test data
#   smp_size <- floor(test_pct*nrow(XY_sep))
#   
#   train_rct_ind <- sample(seq_len(nrow(XY_sep)), size = smp_size)
#   # Which rectangles contain the test data:
#   test_rct <- XY_sep[train_rct_ind, ]
#   # to the test_rct, add the label that it is test data
#   test_rct <- mutate(test_rct, isTestData=1)
#   
#   # Designate as test data all entries in the dataset
#   # that are in the chosel rectangles
#   
#   Xmin <- min(XCoord)
#   Ymin <- min(YCoord)
#   
#   # Update the local data frame with corresponding x and y rectangle separators
#   # to the south and east of each observation
#   df <- df %>%
#     mutate(X_sep = Xmin + floor((XCoord-Xmin)/X_width) * X_width)%>%
#     mutate(Y_sep = Ymin + floor((YCoord -  Ymin)/Y_width) * Y_width)
#   
#   
#   # Combine the local data frame with indicator for test data
#   df <- merge(df, test_rct, by=c("X_sep", "Y_sep"), all.x = TRUE)
#   
#   
#   # Make sure that the indicator for test data is 1 if yes, 0 if no
#   df <- df %>%
#     mutate(isTestData = ifelse(is.na(isTestData),0,1))
#   
#   # return the data frame, with x and y coordinates and the
#   # test data indicator
#   df <- df %>%
#     select(XCoord, YCoord, isTestData)
#   return(df)
# }



# FUNCTION: kfold_rectangular_partitions
# RETURNS:  a data frame with x and y coordinates, and a column
#   with numbers 1, 2, ..., k approximately equally partitioned, 
#   where each value is spatially grouped
#   into rectangles.
# PARAMETERS:
#   XCoord, YCoord: x and y coordinates of the observations
#   num_rectangles: desired number of rectangles
#   k : number of partitions
kfold_rectangular_partitions <- function(
  XCoord, YCoord, 
  num_rectangles=1000, 
  k=12){
  
  # Put the coordinates in a local data frame
  df <- data.frame(XCoord, YCoord)
  
  # resulting side widths of each rectangle
  X_width <- (max(XCoord)-min(XCoord))/ceiling(sqrt(num_rectangles)) 
  Y_width <- (max(YCoord)-min(YCoord))/ceiling(sqrt(num_rectangles)) 
  
  
  # X and Y coordinate markers that separate the rectangles
  X_sep <- seq(min(XCoord), 
               max(XCoord)-X_width, by=X_width)
  Y_sep <- seq(min(YCoord), 
               max(YCoord)-Y_width, by=Y_width)
  
  XY_sep <- crossing(X_sep, Y_sep)
  
  # Randomly split the rectangles into k folds
  # Shuffle the rectangles
  XY_sep <- XY_sep[sample(nrow(XY_sep)), ]
  # create k equally sized folds
  XY_sep$folds <- cut(seq(1, nrow(XY_sep)), breaks = k, labels = FALSE)
  
  # # randomly select which of the rectangles will be in test data
  # smp_size <- floor(test_pct*nrow(XY_sep))
  # 
  # train_rct_ind <- sample(seq_len(nrow(XY_sep)), size = smp_size)
  # # Which rectangles contain the test data:
  # test_rct <- XY_sep[train_rct_ind, ]
  # # to the test_rct, add the label that it is test data
  # test_rct <- mutate(test_rct, isTestData=1)
  # 
  # Designate as test data all entries in the dataset
  # that are in the chosel rectangles
  
  Xmin <- min(XCoord)
  Ymin <- min(YCoord)
  
  # Update the local data frame with corresponding x and y rectangle separators
  # to the south and east of each observation
  df <- df %>%
    mutate(X_sep = Xmin + floor((XCoord-Xmin)/X_width) * X_width)%>%
    mutate(Y_sep = Ymin + floor((YCoord -  Ymin)/Y_width) * Y_width)
  
  
  # Combine the local data frame with indicator for test data
  df <- merge(df, XY_sep, by=c("X_sep", "Y_sep"), all.x = TRUE)
  
  # There may be some small number of observations that are unassigned to a fold.
  # Assign those to fold 1.
  df <- df %>%
    mutate(folds = ifelse(is.na(folds), 1, folds))
  
  
  # return the data frame, with x and y coordinates and the
  # test data indicator
  df <- df %>%
    select(XCoord, YCoord, folds)
  return(df)
}

#####################################################################

#######################################################
# Prep the data
#######################################################

# Load the data
load("../../Data/permafrost_df FNSB CWPP.RData")

# Narrow the dataframe to only the predictive factors we want to have in the simulator,
# plus the vegetation category labels and x-y coordinates.

comb_df <- comb_df %>%
  mutate(ice_wedge = ifelse(thaw_feature == "ice wedge", 1, 0))%>%
  mutate(ice_wedge = as.factor(ice_wedge)) %>%
  select(XCoord, YCoord,  # x-y coordinates
         which_extent,    # ground ice, thaw feature, or other CWPP region
         ground_ice_3,    # labels to extrapolate
         ice_wedge,       # whether there is an "ice wedge" thaw feature.  Available only in thaw feature extent.    
         Tanana_floodplain,         # indicator for the river Tanana floodplain
         ELEV, Local_Elev, SLOPE,   # Relevant topography.  Will need to decide if elevation (ELEV) remains relevant.
         S_TEMP_FNSB_1996TO2005, W_TEMP_FNSB_1996TO2005,  # average decadal temperature
         isWetland,                 # wetland indicator
         ABOVE_1984_adj             # vegetation type
  ) %>%
  rename(temp_summer = S_TEMP_FNSB_1996TO2005) %>%
  rename(temp_winter = W_TEMP_FNSB_1996TO2005)


# Filter only the ground ice extent, and remove which_extent
df <- comb_df %>%
  filter(which_extent == "ground ice") %>%
  select(-which_extent) %>%
  filter(!ABOVE_1984_adj == "Water") %>%
  mutate(ABOVE_1984_adj = droplevels(ABOVE_1984_adj))

rm(comb_df)

# Select only observations with complete records
df <- df[complete.cases(df),]

# Note that ground_ice_3 is NA where there is water.  Ultimately, extrapolate to
# places where there is no water recorded (use ABOVE_1984_adj "Water" category)


folds_df <- kfold_rectangular_partitions(df$XCoord, df$YCoord,
                                         num_rectangles = 1000, k=12)

# # Split the data into training and test sets
# split_df <- rectangular_partitions(df$XCoord, df$YCoord,
#                                    num_rectangles = 1000)

df <- merge(df, folds_df, by=c("XCoord", "YCoord"), all.x = TRUE)

rm(folds_df)


# Visualize the folds, see if they are in rectangles

if (!dir.exists("./Graphs")) {dir.create("./Graphs")}


df  %>%
  mutate(folds=as.factor(folds))%>%
  ggplot(aes(x=XCoord,
             y=YCoord)) +
  geom_raster(aes(fill=folds)) 

ggsave(file="./Graphs/ground ice extent folds.png", height = 5, width = 7)


df <- df %>%
  mutate(isTestData = ifelse(folds >= 10, 1, 0))

# Visualize the folds, see if they are in rectangles
plt <- df  %>%
  mutate(Test_or_Train = ifelse(isTestData==1, "Test", "Train"))%>%
  mutate(Test_or_Train = as.factor(Test_or_Train)) %>%
  ggplot(aes(x=XCoord,
             y=YCoord)) +
  geom_raster(aes(fill=Test_or_Train)) + 
  theme(legend.title = element_blank())


ggsave(plt, file="./Graphs/Test and training data split.png", height = 5, width = 7)

rm(plt)



# Make sure that all the predictive factors are either
# numeric, or of type "factor".  

# 'data.frame':	853870 obs. of  14 variables:
# $ XCoord           : num  282586 282586 282586 282586 282586 ...
# $ YCoord           : num  1683508 1683538 1683568 1683598 1683628 ...
# $ ground_ice_3     : Factor w/ 3 levels "Low","Medium",..: 1 1 1 1 1 1 1 1 1 1 ...
# $ ice_wedge        : Factor w/ 2 levels "0","1": 1 1 1 1 1 1 1 1 1 1 ...
# $ Tanana_floodplain: Factor w/ 2 levels "0","1": 1 1 1 1 1 1 1 1 1 1 ...
# $ ELEV             : num  393 385 378 371 363 ...
# $ Local_Elev       : num  175 168 163 157 151 ...
# $ SLOPE            : num  19.3 18.9 18.5 18 15.9 ...
# $ temp_summer      : num  14.6 14.6 14.6 14.6 14.6 ...
# $ temp_winter      : num  -14.5 -14.5 -14.5 -14.5 -14.5 ...
# $ isWetland        : Factor w/ 2 levels "0","1": 2 1 1 2 1 1 1 1 1 1 ...
# $ ABOVE_1984_adj   : Factor w/ 15 levels "Shallows","Bog",..: 11 15 15 11 11 15 15 15 15 15 ...
# $ folds            : int  8 8 8 8 8 8 8 8 8 8 ...
# $ isTestData       : num  0 0 0 0 0 0 0 0 0 0 ...


# Save for future.
save(df, file = "train test folds df.RData")

