package Ensemble_Model;

import java.util.ArrayList;

public class Quadrup {
	
	public int uID = 0;
	
	public int sID = 0;
	
	public int tID = 0;
	
	public double value = 0;
	
	public double dRatingHat_RMSE = 0;
	public double dRatingHat_MAE = 0;  
	public double dRatingHat_NSE = 0;  
	
	public ArrayList<Double> dRatingHats_RMSE = new ArrayList<>();
	public ArrayList<Double> dRatingHats_MAE = new ArrayList<>();  
	public ArrayList<Double> dRatingHats_NSE = new ArrayList<>();  
	
	public double dWeight_RMSE = 0;
	public double dWeight_MAE = 0; 
	public double dWeight_NSE = 0; 

}
