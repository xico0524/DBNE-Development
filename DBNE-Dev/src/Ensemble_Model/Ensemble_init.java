package Ensemble_Model;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.StringTokenizer;

public class Ensemble_init{

	public static String inputTrainFile = "./data/training.txt"; 
	public static String inputValidFile = "./data/validation.txt";  
	public static String inputTestFile = "./data/testing.txt";  
	
	public static String separator = ":";
	
	public static ArrayList<Quadrup> trainData = new ArrayList<Quadrup>();
	public static ArrayList<Quadrup> validData = new ArrayList<Quadrup>();
	public static ArrayList<Quadrup> testData = new ArrayList<Quadrup>();
	
	public static int uNum = 0;
	public static int sNum = 0;
	public static int tNum = 0;

	public static double max_Num = 0; 
	public static double min_Num = 0; 
	public static double nor_pNum = 0; 
	public static double max_nor_Num = 0; 
	public static double min_nor_Num = 0;
	
	public static int r = 20;
	
	public static int trainDataNum = 0;
	public static int validDataNum = 0;
	public static int testDataNum = 0;

	public static int delayCount = 10;

	public static void initData(String inputFile,ArrayList<Quadrup> data, int T)throws IOException
	{
		
		File input = new File(inputFile);
		BufferedReader in = new BufferedReader(new FileReader(input));
		String inTemp;

		while((inTemp = in.readLine()) != null ) {
			StringTokenizer st = new StringTokenizer(inTemp,separator);

			String iTemp = null;
			if(st.hasMoreTokens())
				iTemp = st.nextToken();
			
			String jTemp = null;
			if(st.hasMoreTokens())
				jTemp = st.nextToken();
			
			String kTemp = null;
			if(st.hasMoreTokens())
				kTemp = st.nextToken();
			
			String tValueTemp = null;
			if(st.hasMoreTokens())
				tValueTemp = st.nextToken();
			
			
			int uID = Integer.valueOf(iTemp);

			double s = Double.valueOf(jTemp);
			int sID = (int) s;

			double t = Double.valueOf(kTemp); 
			int tID = (int) t;
			
			double Value = Double.valueOf(tValueTemp);

			uNum = (uNum > uID) ? uNum : uID;
			sNum = (sNum > sID) ? sNum : sID;
			tNum = (tNum > tID) ? tNum : tID;

			max_Num = (max_Num > Value) ? max_Num : Value;
			min_Num = (min_Num < Value) ? min_Num : Value;	
			
			if(T==0) {
				trainDataNum++;
			}
			else {
				if(T==1) {
					validDataNum++;
				}
				else {
					testDataNum++;
				}
			}

			nor_pNum = Math.log(Value+1);

			max_nor_Num = (max_nor_Num > nor_pNum) ? max_nor_Num : nor_pNum;
			min_nor_Num = (min_nor_Num < nor_pNum) ? min_nor_Num : nor_pNum;
			
			Quadrup qtemp = new Quadrup();
			qtemp.uID = uID;
			qtemp.sID = sID;
			qtemp.tID = tID;

			qtemp.value = nor_pNum;

			data.add(qtemp);
		}
		
		if(T==0)
		{
			System.out.println(trainDataNum);
		}
		if (T==1)
		{
			System.out.println(validDataNum);
		}
		if (T==2)
		{
			System.out.println(testDataNum);
		}
		
		in.close();
	}
	

	public static Map<Integer, ArrayList<RTuple>> USlice = null;
	public static Map<Integer, ArrayList<RTuple>> SSlice = null;
	public static Map<Integer, ArrayList<RTuple>> TSlice = null;

	public static void partSlice()
	{
		USlice = new HashMap<Integer,ArrayList<RTuple>>();
		SSlice = new HashMap<Integer,ArrayList<RTuple>>();
		TSlice = new HashMap<Integer,ArrayList<RTuple>>();
		
		for (Quadrup slice1: trainData)
		{
			if(USlice.containsKey(Integer.valueOf(slice1.uID)))
			{
				RTuple rtemp = new RTuple();
				rtemp.rowID = slice1.sID;
				rtemp.colID = slice1.tID;
				rtemp.mvalue = slice1.value;
				
				USlice.get(Integer.valueOf(slice1.uID)).add(rtemp);
			
			}else {
				ArrayList<RTuple> uSlice = new ArrayList<RTuple>();
				RTuple rtemp = new RTuple();
				rtemp.rowID = slice1.sID;
				rtemp.colID = slice1.tID;
				rtemp.mvalue = slice1.value;
				uSlice.add(rtemp);

				USlice.put(Integer.valueOf(slice1.uID),uSlice);	
			}
			
			if(SSlice.containsKey(Integer.valueOf(slice1.sID)))
			{
				RTuple rtemp = new RTuple();
				rtemp.rowID = slice1.uID;
				rtemp.colID = slice1.tID;
				rtemp.mvalue = slice1.value;
				SSlice.get(Integer.valueOf(slice1.sID)).add(rtemp);
				
			}else {
				ArrayList<RTuple> sSlice = new ArrayList<RTuple>();
				RTuple rtemp = new RTuple();
				rtemp.rowID = slice1.uID;
				rtemp.colID = slice1.tID;
				rtemp.mvalue = slice1.value;
				sSlice.add(rtemp);
				SSlice.put(Integer.valueOf(slice1.sID),sSlice);	
			}
			
			
			if(TSlice.containsKey(Integer.valueOf(slice1.tID)))
			{
				RTuple rtemp = new RTuple();
				rtemp.rowID = slice1.uID;
				rtemp.colID = slice1.sID;
				rtemp.mvalue = slice1.value;
				TSlice.get(Integer.valueOf(slice1.tID)).add(rtemp);
				
			}else {
				ArrayList<RTuple> tSlice = new ArrayList<RTuple>();
				RTuple rtemp = new RTuple();
				rtemp.rowID = slice1.uID;
				rtemp.colID = slice1.sID;
				rtemp.mvalue = slice1.value;
				tSlice.add(rtemp);
				TSlice.put(Integer.valueOf(slice1.tID),tSlice);	
			}
			
		}
	}

	public static void init_Weight() 
	{
		for(Quadrup temp : trainData) 
		{
			temp.dWeight_RMSE = 1.0 / trainDataNum;
			temp.dWeight_MAE = 1.0 / trainDataNum;
			temp.dWeight_NSE = 1.0 / trainDataNum;
		}
	}

	public static double global_miu_valid = 0;
	public static double global_miu_test = 0;
	
	public static void compute_ave() 
	{
		double miu_valid = 0;
		for(Quadrup valid : validData)
		{
			miu_valid += valid.value;
		}
		global_miu_valid = miu_valid / validDataNum;
		
		double miu_test = 0;
		for(Quadrup test : testData) 
		{
			miu_test += test.value;
		}
		global_miu_test = miu_test / testDataNum;
	}

	public static void main(String[] args)throws NumberFormatException,IOException,InterruptedException {
		
		System.out.println("Start");
		System.out.println("************************************");
		double all_start = System.currentTimeMillis();
		
		initData(inputTrainFile,trainData, 0);
		initData(inputValidFile,validData, 1);
		initData(inputTestFile,testData, 2);	
		partSlice();
		init_Weight();
		
		compute_ave();
		
		double sum_omega_RMSE = 0;
		double sum_omega_MAE = 0;
		double sum_omega_NSE = 0; 

		M1_SLBN M1 = new M1_SLBN();		
		double[] M1_result = M1.boosting();
		sum_omega_RMSE += M1_result[0];
		sum_omega_MAE += M1_result[1];
		sum_omega_NSE += M1_result[2];

		M2_ELBN M2 = new M2_ELBN();
		double[] M2_result = M2.boosting();
		sum_omega_RMSE += M2_result[0];
		sum_omega_MAE += M2_result[1];
		sum_omega_NSE += M2_result[2];

		M3_WPBN M3 = new M3_WPBN();
		double[] M3_result = M3.boosting();
		sum_omega_RMSE += M3_result[0];
		sum_omega_MAE += M3_result[1];
		sum_omega_NSE += M3_result[2];
		
		M4_TBN M4 = new M4_TBN();
		double[] M4_result = M4.boosting();
		sum_omega_RMSE += M4_result[0];
		sum_omega_MAE += M4_result[1];
		sum_omega_NSE += M4_result[2];
		
		M5_PLBN M5 = new M5_PLBN();
		double[] M5_result = M5.boosting();
		sum_omega_RMSE += M5_result[0];
		sum_omega_MAE += M5_result[1];
		sum_omega_NSE += M5_result[2];

		M6_PLogBN M6 = new M6_PLogBN();
		double[] M6_result = M6.boosting();
		sum_omega_RMSE += M6_result[0];
		sum_omega_MAE += M6_result[1];
		sum_omega_NSE += M6_result[2];

		M7_PSBN M7= new M7_PSBN();
		double[] M7_result = M7.boosting();
		sum_omega_RMSE += M7_result[0];
		sum_omega_MAE += M7_result[1];
		sum_omega_NSE += M7_result[2];
		
		double sumRMSE = 0, sumMAE = 0;
		double sumNSEUp = 0, sumNSEDown = 0;
		for(Quadrup test : testData) 
		{
			test.dRatingHat_RMSE = test.dRatingHat_RMSE / sum_omega_RMSE;
			test.dRatingHat_MAE = test.dRatingHat_MAE / sum_omega_MAE;
			test.dRatingHat_NSE = test.dRatingHat_NSE / sum_omega_NSE;
			
			sumRMSE += Math.pow(test.value - test.dRatingHat_RMSE, 2);
			sumMAE += Math.abs(test.value - test.dRatingHat_MAE);
			
			sumNSEUp += Math.pow(test.value - test.dRatingHat_NSE, 2);
			sumNSEDown += Math.pow(test.value - global_miu_test, 2);
		}
		
		double rmse_avg = Math.sqrt(sumRMSE / testDataNum);
		double mae_avg = sumMAE / testDataNum;
		double nse_avg = 1 - (sumNSEUp / sumNSEDown);
		
		System.out.println("\nEnsemble: RMSE:"+rmse_avg
				+",MAE:"+mae_avg
				+",NSE:"+nse_avg);
		
		
		double all_end = System.currentTimeMillis();
		System.out.println("Total time cost:"+(all_end-all_start)/1000+"Seconds");
		System.out.println("************************************");
		System.out.println("End");

	}
}
