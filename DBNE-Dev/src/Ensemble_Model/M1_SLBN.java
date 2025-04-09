package Ensemble_Model;

import java.util.ArrayList;
import java.util.Arrays; 
import java.util.Random;

public class M1_SLBN extends Ensemble_init{

	public double lambda = 1.0E-3; 
	public double lambda_b = 1.0E-3;
	
	public double[][] U;
	public double[][] S;
	public double[][] T;
	
	public double[]	A;
	public double[] B;
	public double[] C;
	
	public double[] everyRoundRMSE;
	public double[] everyRoundMAE;
	public double[] everyRoundNSE;
	
	public int trainingRound = 1000;
	public int convergenceRound = 1000;
	
	public boolean flagRMSE = false;
	public boolean flagMAE = false; 
	public boolean flagNSE = false;
	
	public double minRMSE = 100;
	public double minMAE = 100; 
	public double minNSE = -100;
	
	public double RMSE_final = 100;
	public double MAE_final = 100;
	public double NSE_final = -100;
	
	public int minRMSERound = 0;
	public int minMAERound = 0;
	public int minNSERound = 0;
	
	public double valid_train_time = 0;
	
	public double randMax = 0.5;
	public double randMin = 4.9E-324;
	public void initUST(int rank)
	{	
		U = new double[uNum+1][rank+1];
		S = new double[sNum+1][rank+1];
		T = new double[tNum+1][rank+1];
		
		Random randomu = new Random();
		for(int i=1; i<=uNum;i++) 
		{
			for(int j=1; j<=rank; j++) 
			{
				U[i][j] = randMin + randomu.nextDouble() * (randMax - randMin);
			}
		}
		
		Random randoms = new Random();
		for(int i=1; i<=sNum; i++) 
		{
			for(int j=1; j<=rank; j++) 
			{
				S[i][j] = randMin + randoms.nextDouble() * (randMax - randMin);
			}
		}
		
		Random randomt = new Random();
		for(int i=1; i<=tNum; i++)
		{
			for(int j=1; j<=rank; j++)
			{
				T[i][j] = randMin + randomt.nextDouble() * (randMax - randMin);
			}
		}
		
	}

	public double biasMax = randMax;
	public double biasMin = randMin;
	
	public void initBias() 
	{
		A = new double[uNum+1];
		B = new double[sNum+1];
		C = new double[tNum+1];
		
		Random randoma = new Random(System.currentTimeMillis());
		for(int i=1; i<=uNum;i++) 
		{
			A[i] = biasMin + randoma.nextDouble() * (biasMax - biasMin);
		}
		
		Random randomb = new Random();
		for(int j=1; j<=sNum; j++) 
		{	
			B[j] = biasMin + randomb.nextDouble() * (biasMax - biasMin);
		}
		
		Random randomc = new Random();
		for(int k=1; k<=tNum; k++) 
		{
			C[k] = biasMin + randomc.nextDouble() * (biasMax - biasMin);
		}
	}

	public double[][] U_update_RMSE;
	public double[][] S_update_RMSE;
	public double[][] T_update_RMSE;
	public double[]	A_update_RMSE; 
	public double[] B_update_RMSE;
	public double[] C_update_RMSE;
	
	public double[][] U_update_MAE;
	public double[][] S_update_MAE;
	public double[][] T_update_MAE;
	public double[]	A_update_MAE; 
	public double[] B_update_MAE;
	public double[] C_update_MAE;
	
	public double[][] U_update_NSE;
	public double[][] S_update_NSE;
	public double[][] T_update_NSE;
	public double[]	A_update_NSE; 
	public double[] B_update_NSE;
	public double[] C_update_NSE;
	
	public void train(int rank)
	{
		U_update_RMSE = new double[uNum+1][rank+1];
		S_update_RMSE = new double[sNum+1][rank+1];
		T_update_RMSE = new double[tNum+1][rank+1];
		A_update_RMSE = new double[uNum+1];
		B_update_RMSE = new double[sNum+1];
		C_update_RMSE = new double[tNum+1];
		
		U_update_MAE = new double[uNum+1][rank+1];
		S_update_MAE = new double[sNum+1][rank+1];
		T_update_MAE = new double[tNum+1][rank+1];
		A_update_MAE = new double[uNum+1];
		B_update_MAE = new double[sNum+1];
		C_update_MAE = new double[tNum+1];
		
		U_update_NSE = new double[uNum+1][rank+1];
		S_update_NSE = new double[sNum+1][rank+1];
		T_update_NSE = new double[tNum+1][rank+1];
		A_update_NSE = new double[uNum+1];
		B_update_NSE = new double[sNum+1];
		C_update_NSE = new double[tNum+1];

		double starttime = System.currentTimeMillis();
		
		everyRoundRMSE = new double[trainingRound+1];
		everyRoundMAE = new double[trainingRound+1];
		everyRoundNSE = new double[trainingRound+1];
		
		minRMSE = 100;
		minMAE =100; 
		minNSE = -100;
		
		minRMSERound = 0;
		minMAERound = 0; 
		minNSERound = 0;

		for(int tr=1; tr<=trainingRound; tr++)
		{
			
			for(int l=1; l<=rank; l++)
			{
				for(int i=1; i<=uNum; i++) 
				{
					double uUp = 0;
					double uDown = 0;
					
					if(USlice.containsKey(Integer.valueOf(i))) 
					{
						ArrayList<RTuple> uSlice = new ArrayList<RTuple>(USlice.get(Integer.valueOf(i)));
						
						for(RTuple utemp: uSlice) 
						{
							uUp += utemp.mvalue * S[utemp.rowID][l] * T[utemp.colID][l];
							
							double uDownTemp = 0;
							for(int r=1; r<=rank; r++) 
							{
								uDownTemp += U[i][r] * S[utemp.rowID][r] * T[utemp.colID][r];
							}
							uDownTemp += A[i] + B[utemp.rowID] + C[utemp.colID];
							
							uDown += uDownTemp * S[utemp.rowID][l] * T[utemp.colID][l];	
							uDown += lambda * U[i][l];
									
						}
					}
					
					uUp = U[i][l] * uUp;
					if(uDown == 0)
						uDown = 4.9E-324;
					
					U[i][l] = uUp / uDown;
				}
			}

			for(int l=1; l<=rank; l++)
			{
				for(int j=1; j<=sNum; j++) 
				{
					double sUp = 0;
					double sDown = 0;
					
					if(SSlice.containsKey(Integer.valueOf(j))) 
					{
						ArrayList<RTuple> sSlice = new ArrayList<RTuple>(SSlice.get(Integer.valueOf(j)));
						
						for(RTuple stemp: sSlice) 
						{
							sUp += stemp.mvalue * U[stemp.rowID][l] * T[stemp.colID][l];
							
							double sDownTemp = 0;
							for(int r=1; r<=rank; r++) 
							{
								sDownTemp += S[j][r] * U[stemp.rowID][r] * T[stemp.colID][r];
							}
							sDownTemp += A[stemp.rowID] + B[j] + C[stemp.colID];
							
							sDown += sDownTemp * U[stemp.rowID][l] * T[stemp.colID][l];	
							sDown += lambda * S[j][l];	
						}
					}
					
					sUp = S[j][l] * sUp;
					if(sDown == 0)
						sDown = 4.9E-324;
					
					S[j][l] = sUp / sDown;
				}
			}

			for(int l=1; l<=rank; l++)
			{
				for(int k=1; k<=tNum; k++) 
				{
					double tUp = 0;
					double tDown = 0;
					
					if(TSlice.containsKey(Integer.valueOf(k))) 
					{
						ArrayList<RTuple> tSlice = new ArrayList<RTuple>(TSlice.get(Integer.valueOf(k)));
						
						for(RTuple ttemp: tSlice) 
						{
							tUp += ttemp.mvalue * U[ttemp.rowID][l] * S[ttemp.colID][l];
							
							double tDownTemp = 0;
							for(int r=1; r<=rank; r++) 
							{
								tDownTemp += T[k][r] * U[ttemp.rowID][r] * S[ttemp.colID][r];
							}
							tDownTemp += A[ttemp.rowID] + B[ttemp.colID] + C[k];
							
							tDown += tDownTemp * U[ttemp.rowID][l] * S[ttemp.colID][l];	
							tDown += lambda * T[k][l];	
						}
					}
					
					tUp = T[k][l] * tUp;
					if(tDown == 0)
						tDown = 4.9E-324;
					
					T[k][l] = tUp / tDown;
				}
			}

			for(int i=1; i<=uNum;i++) 
			{
				double aUp = 0;
				double aDown = 0;

				if(USlice.containsKey(Integer.valueOf(i)))
				{
					ArrayList<RTuple> uSlice = new ArrayList<RTuple>(USlice.get(Integer.valueOf(i)));
					
					for(RTuple utemp: uSlice) 
					{
						aUp += utemp.mvalue;
						
						for(int r=1; r<=rank; r++)
						{
							aDown += U[i][r] * S[utemp.rowID][r] * T[utemp.colID][r];
						}
						
						aDown += A[i] + B[utemp.rowID] + C[utemp.colID];
						aDown += lambda_b * A[i];
					}

					if(aDown == 0)
						aDown = 4.9E-324;
					aUp = A[i] * aUp;
					A[i] = aUp / aDown;
				}	
			}
			
			for(int j=1; j<=sNum;j++) 
			{
				double bUp = 0;
				double bDown = 0;

				if(SSlice.containsKey(Integer.valueOf(j)))
				{
					ArrayList<RTuple> sSlice = new ArrayList<RTuple>(SSlice.get(Integer.valueOf(j)));
					
					for(RTuple stemp: sSlice) 
					{
						bUp += stemp.mvalue;
						
						for(int r=1; r<=rank; r++)
						{
							bDown += U[stemp.rowID][r] * S[j][r] * T[stemp.colID][r];
						}
						
						bDown += A[stemp.rowID] + B[j] + C[stemp.colID];
						bDown += lambda_b * B[j];
					}
					
					if(bDown == 0)
						bDown = 4.9E-324;
					bUp = B[j] * bUp;
					B[j] = bUp / bDown;
				}	
			}

			for(int k=1; k<=tNum;k++) 
			{
				double cUp = 0;
				double cDown = 0;

				if(TSlice.containsKey(Integer.valueOf(k)))
				{
					ArrayList<RTuple> tSlice = new ArrayList<RTuple>(TSlice.get(Integer.valueOf(k)));
					
					for(RTuple ttemp: tSlice) 
					{
						cUp += ttemp.mvalue;
						
						for(int r=1; r<=rank; r++)
						{
							cDown += U[ttemp.rowID][r] * S[ttemp.colID][r] * T[k][r];
						}
						
						cDown += A[ttemp.rowID] + B[ttemp.colID] + C[k];
						cDown += lambda_b * C[k];
					}
					
					if(cDown == 0)
						cDown = 4.9E-324;
					cUp = C[k] * cUp;
					C[k] = cUp / cDown;
				}	
			}

			double RMSEUp = 0; 
			double MAEUp = 0;
			
			double NSEUp = 0;
			double NSEDown = 0;
			
			for(Quadrup valid : validData) 
			{
				double ytemp = 0;
				for(int yr=1; yr<=rank; yr++) 
				{
					ytemp += U[valid.uID][yr] * S[valid.sID][yr] * T[valid.tID][yr];
				}
				ytemp += A[valid.uID] + B[valid.sID] + C[valid.tID];
				
				RMSEUp += Math.pow(valid.value - ytemp, 2);
				MAEUp += Math.abs(valid.value - ytemp);
				
				NSEUp += Math.pow(valid.value - ytemp, 2);
				NSEDown += Math.pow(valid.value - global_miu_valid, 2);	
				
			}
			
			everyRoundRMSE[tr] = Math.sqrt(RMSEUp/validDataNum);
			everyRoundMAE[tr] = MAEUp / validDataNum;
			everyRoundNSE[tr] = 1 - ( NSEUp / NSEDown );

			if((Math.abs(everyRoundRMSE[tr]-minRMSE)>=0.0001) && (everyRoundRMSE[tr]<minRMSE))
			{
				minRMSE = everyRoundRMSE[tr];
				minRMSERound = tr;
				
				for(int i=1; i<=uNum; i++) 
				{
					A_update_RMSE[i] = A[i];
					for(int r=1; r<=rank; r++) 
					{
						U_update_RMSE[i][r] = U[i][r];
					}
				}
				
				for(int j=1; j<=sNum; j++) 
				{
					B_update_RMSE[j] = B[j];
					for(int r=1; r<=rank; r++) 
					{
						S_update_RMSE[j][r] = S[j][r];
					}
				}
				
				for(int k=1; k<=tNum; k++) 
				{
					C_update_RMSE[k] = C[k];
					for(int r=1; r<=rank; r++) 
					{
						T_update_RMSE[k][r] = T[k][r];
					}
				}
			}
			
			else
			{
				if((tr - minRMSERound) >= delayCount) 
				{
					flagRMSE = true;
					if(flagMAE && flagNSE) 
					{
						convergenceRound = tr;
						break;
					}
				}
			}

			if((Math.abs(everyRoundMAE[tr]-minMAE)>=0.0001) && (everyRoundMAE[tr] < minMAE))
			{
				minMAE = everyRoundMAE[tr];
				minMAERound = tr;
				
				for(int i=1; i<=uNum; i++) 
				{
					A_update_MAE[i] = A[i];
					for(int r=1; r<=rank; r++) 
					{
						U_update_MAE[i][r] = U[i][r];
					}
				}
				
				for(int j=1; j<=sNum; j++) 
				{
					B_update_MAE[j] = B[j];
					for(int r=1; r<=rank; r++) 
					{
						S_update_MAE[j][r] = S[j][r];
					}
				}
				
				for(int k=1; k<=tNum; k++) 
				{
					C_update_MAE[k] = C[k];
					for(int r=1; r<=rank; r++) 
					{
						T_update_MAE[k][r] = T[k][r];
					}
				}
			}
			
			else
			{
				if((tr - minMAERound) >= delayCount) 
				{
					flagMAE = true;
					if(flagRMSE && flagNSE)
					{
						convergenceRound = tr;
						break;
					}
				}
			}

			if((Math.abs(minNSE-everyRoundNSE[tr])>=0.0001) && (everyRoundNSE[tr] > minNSE))
			{
				minNSE = everyRoundNSE[tr];
				minNSERound = tr;
				
				for(int i=1; i<=uNum; i++) 
				{
					A_update_NSE[i] = A[i];
					for(int r=1; r<=rank; r++) 
					{
						U_update_NSE[i][r] = U[i][r];
					}
				}
				
				for(int j=1; j<=sNum; j++) 
				{
					B_update_NSE[j] = B[j];
					for(int r=1; r<=rank; r++) 
					{
						S_update_NSE[j][r] = S[j][r];
					}
				}
				
				for(int k=1; k<=tNum; k++) 
				{
					C_update_NSE[k] = C[k];
					for(int r=1; r<=rank; r++) 
					{
						T_update_NSE[k][r] = T[k][r];
					}
				}
			}
			
			else
			{
				if((tr - minNSERound) >= delayCount) 
				{
					flagNSE = true;
					if(flagRMSE && flagMAE)
					{
						convergenceRound = tr;
						break;
					}
				}
			}
		}
		
		System.out.println("\nM1 Validation Convergence Round: "+convergenceRound);
		System.out.println("RMSE: "+minRMSE+", Round: "+minRMSERound);
		System.out.println("MAE: "+minMAE+", Round: "+minMAERound);
		System.out.println("NSE: "+minNSE+", Round: "+minNSERound);

		double RMSEUp_final = 0; 
		double MAEUp_final = 0;
		
		double NSEUp_final = 0;
		double NSEDown_final = 0;
		
		for (Quadrup test: testData)
		{
			double ytemp_RMSE = 0;
			double ytemp_MAE = 0;
			double ytemp_NSE = 0;
			
			for(int yr=1; yr<=rank; yr++)
			{
				ytemp_RMSE += U_update_RMSE[test.uID][yr] * S_update_RMSE[test.sID][yr] * T_update_RMSE[test.tID][yr];
				ytemp_MAE += U_update_MAE[test.uID][yr] * S_update_MAE[test.sID][yr] * T_update_MAE[test.tID][yr]; 
				ytemp_NSE += U_update_NSE[test.uID][yr] * S_update_NSE[test.sID][yr] * T_update_NSE[test.tID][yr];
			}
			ytemp_RMSE += A_update_RMSE[test.uID] + B_update_RMSE[test.sID] + C_update_RMSE[test.tID];
			ytemp_MAE += A_update_MAE[test.uID] + B_update_MAE[test.sID] + C_update_MAE[test.tID]; 
			ytemp_NSE += A_update_NSE[test.uID] + B_update_NSE[test.sID] + C_update_NSE[test.tID]; 
			
			RMSEUp_final += Math.pow(test.value- ytemp_RMSE, 2);
			MAEUp_final += Math.abs(test.value - ytemp_MAE);	
			
			NSEUp_final += Math.pow(test.value - ytemp_NSE, 2);
			NSEDown_final += Math.pow(test.value - global_miu_test, 2);
			
		}
		
		RMSE_final = Math.sqrt(RMSEUp_final / testDataNum);
		MAE_final = MAEUp_final / testDataNum;
		NSE_final = 1 - (NSEUp_final / NSEDown_final);
		
		System.out.println("Testing RMSE: " + RMSE_final);
		System.out.println("MAE: " + MAE_final);
		System.out.println("NSE: " + NSE_final);
		
		double endtime = System.currentTimeMillis();
		
		valid_train_time = (endtime-starttime)/1000;
		System.out.println("Training Time: "+ valid_train_time +" Seconds");
		
	}
	
	public double[] boosting() 
	{
		initUST(r);
		initBias();
		train(r);

		double omega_RMSE = 0;
		double omega_MAE = 0; 
		double omega_NSE = 0; 
		
		double ratingHat_RMSE;
		double ratingHat_MAE; 
		double ratingHat_NSE; 
		
		double err_RMSE = 0;
		double err_MAE = 0;
		double err_NSE = 0;
		
		double miu_RMSE = 0;
		double miu_MAE = 0;
		double miu_NSE = 0;
		
		double sigma_RMSE = 0;
		double sigma_MAE = 0; 
		double sigma_NSE = 0; 
		
		double phi_RMSE = 0;
		double phi_MAE = 0;
		double phi_NSE = 0;
		
		double Z_RMSE = 0;
		double Z_MAE = 0; 
		double Z_NSE = 0; 
		
		double a = 1;
		
		for(Quadrup temp : trainData) 
		{
			double ytemp_RMSE = 0;
			double ytemp_MAE = 0;
			double ytemp_NSE = 0;
			
			for(int yr=1; yr<=r; yr++) 
			{
				ytemp_RMSE += U_update_RMSE[temp.uID][yr] * S_update_RMSE[temp.sID][yr] * T_update_RMSE[temp.tID][yr];
				ytemp_MAE += U_update_MAE[temp.uID][yr] * S_update_MAE[temp.sID][yr] * T_update_MAE[temp.tID][yr];
				ytemp_NSE += U_update_NSE[temp.uID][yr] * S_update_NSE[temp.sID][yr] * T_update_NSE[temp.tID][yr];
			}
			ytemp_RMSE += A_update_RMSE[temp.uID] + B_update_RMSE[temp.sID] + C_update_RMSE[temp.tID];
			ytemp_MAE += A_update_MAE[temp.uID] + B_update_MAE[temp.sID] + C_update_MAE[temp.tID];
			ytemp_NSE += A_update_NSE[temp.uID] + B_update_NSE[temp.sID] + C_update_NSE[temp.tID];
			
			ratingHat_RMSE = ytemp_RMSE;
			ratingHat_MAE = ytemp_MAE;
			ratingHat_NSE = ytemp_NSE;
			
			err_RMSE = Math.abs(temp.value - ratingHat_RMSE);
			err_MAE = Math.abs(temp.value - ratingHat_MAE);
			err_NSE = Math.abs(temp.value - ratingHat_NSE);
			
			miu_RMSE += err_RMSE;
			miu_MAE += err_MAE;
			miu_NSE += err_NSE;
			
			sigma_RMSE += Math.pow(err_RMSE, 2);	
			sigma_MAE += Math.pow(err_MAE, 2);	
			sigma_NSE += Math.pow(err_NSE, 2);	
			
		}
		
		miu_RMSE = miu_RMSE / trainDataNum;
		miu_MAE = miu_MAE / trainDataNum; 
		miu_NSE = miu_NSE / trainDataNum; 
		
		sigma_RMSE = Math.sqrt((sigma_RMSE / trainDataNum) - Math.pow(miu_RMSE, 2));
		sigma_MAE = Math.sqrt((sigma_MAE / trainDataNum) - Math.pow(miu_MAE, 2)); 
		sigma_NSE = Math.sqrt((sigma_NSE / trainDataNum) - Math.pow(miu_NSE, 2)); 
		
		for(Quadrup temp : trainData) 
		{
			double ytemp_RMSE = 0;
			double ytemp_MAE = 0;
			double ytemp_NSE = 0;
			
			for(int yr=1; yr<=r; yr++) 
			{
				ytemp_RMSE += U_update_RMSE[temp.uID][yr] * S_update_RMSE[temp.sID][yr] * T_update_RMSE[temp.tID][yr];
				ytemp_MAE += U_update_MAE[temp.uID][yr] * S_update_MAE[temp.sID][yr] * T_update_MAE[temp.tID][yr];
				ytemp_NSE += U_update_NSE[temp.uID][yr] * S_update_NSE[temp.sID][yr] * T_update_NSE[temp.tID][yr];
			}
			ytemp_RMSE += A_update_RMSE[temp.uID] + B_update_RMSE[temp.sID] + C_update_RMSE[temp.tID];
			ytemp_MAE += A_update_MAE[temp.uID] + B_update_MAE[temp.sID] + C_update_MAE[temp.tID];
			ytemp_NSE += A_update_NSE[temp.uID] + B_update_NSE[temp.sID] + C_update_NSE[temp.tID];
			
			ratingHat_RMSE = ytemp_RMSE;
			ratingHat_MAE = ytemp_MAE;
			ratingHat_NSE = ytemp_NSE;
			
			err_RMSE = Math.abs(temp.value - ratingHat_RMSE);
			err_MAE = Math.abs(temp.value - ratingHat_MAE);
			err_NSE = Math.abs(temp.value - ratingHat_NSE);
			
			if((err_RMSE-miu_RMSE) > a*sigma_RMSE) 
			{
				phi_RMSE += temp.dWeight_RMSE;
			}		
			
			if((err_MAE-miu_MAE) > a*sigma_MAE) 
			{
				phi_MAE += temp.dWeight_MAE;
			}
			
			if((err_NSE-miu_NSE) > a*sigma_NSE) 
			{
				phi_NSE += temp.dWeight_NSE;
			}

		}
		
		for(Quadrup temp : trainData) 
		{
			double ytemp_RMSE = 0;
			double ytemp_MAE = 0;
			double ytemp_NSE = 0;
			
			for(int yr=1; yr<=r; yr++) 
			{
				ytemp_RMSE += U_update_RMSE[temp.uID][yr] * S_update_RMSE[temp.sID][yr] * T_update_RMSE[temp.tID][yr];
				ytemp_MAE += U_update_MAE[temp.uID][yr] * S_update_MAE[temp.sID][yr] * T_update_MAE[temp.tID][yr];
				ytemp_NSE += U_update_NSE[temp.uID][yr] * S_update_NSE[temp.sID][yr] * T_update_NSE[temp.tID][yr];
			}
			ytemp_RMSE += A_update_RMSE[temp.uID] + B_update_RMSE[temp.sID] + C_update_RMSE[temp.tID];
			ytemp_MAE += A_update_MAE[temp.uID] + B_update_MAE[temp.sID] + C_update_MAE[temp.tID];
			ytemp_NSE += A_update_NSE[temp.uID] + B_update_NSE[temp.sID] + C_update_NSE[temp.tID];
			
			ratingHat_RMSE = ytemp_RMSE;
			ratingHat_MAE = ytemp_MAE;
			ratingHat_NSE = ytemp_NSE;
			
			err_RMSE = Math.abs(temp.value - ratingHat_RMSE);
			err_MAE = Math.abs(temp.value - ratingHat_MAE);
			err_NSE = Math.abs(temp.value - ratingHat_NSE);
			
			if((err_RMSE-miu_RMSE) <= a*sigma_RMSE) 
			{
				temp.dWeight_RMSE = temp.dWeight_RMSE * phi_RMSE;
			}
			Z_RMSE += temp.dWeight_RMSE;		
			
			if((err_MAE-miu_MAE) <= a*sigma_MAE) 
			{
				temp.dWeight_MAE = temp.dWeight_MAE * phi_MAE;
			}
			Z_MAE += temp.dWeight_MAE;	
			
			if((err_NSE-miu_NSE) <= a*sigma_NSE) 
			{
				temp.dWeight_NSE = temp.dWeight_NSE * phi_NSE;
			}
			Z_NSE += temp.dWeight_NSE;
			
		}
		
		for(Quadrup temp : trainData) 
		{
			temp.dWeight_RMSE = temp.dWeight_RMSE / Z_RMSE;
			temp.dWeight_MAE = temp.dWeight_MAE / Z_MAE;
			temp.dWeight_NSE = temp.dWeight_NSE / Z_NSE;

		}
		
		omega_RMSE = Math.log(1 / phi_RMSE);
		omega_MAE = Math.log(1 / phi_MAE); 
		omega_NSE = Math.log(1 / phi_NSE); 
		
		double[] result_M1_omega = new double[3];
		result_M1_omega[0] = omega_RMSE;
		result_M1_omega[1] = omega_MAE;
		result_M1_omega[2] = omega_NSE;
		
		System.out.println("Weight: "+ Arrays.toString(result_M1_omega));

		for(Quadrup test : testData) 
		{
			double ytemp_RMSE = 0;
			double ytemp_MAE = 0;
			double ytemp_NSE = 0;
			
			for(int yr=1; yr<=r; yr++)
			{
				ytemp_RMSE += U_update_RMSE[test.uID][yr] * S_update_RMSE[test.sID][yr] * T_update_RMSE[test.tID][yr];
				ytemp_MAE += U_update_MAE[test.uID][yr] * S_update_MAE[test.sID][yr] * T_update_MAE[test.tID][yr]; 
				ytemp_NSE += U_update_NSE[test.uID][yr] * S_update_NSE[test.sID][yr] * T_update_NSE[test.tID][yr]; 
			}
			ytemp_RMSE += A_update_RMSE[test.uID] + B_update_RMSE[test.sID] + C_update_RMSE[test.tID];
			ytemp_MAE += A_update_MAE[test.uID] + B_update_MAE[test.sID] + C_update_MAE[test.tID]; 
			ytemp_NSE += A_update_NSE[test.uID] + B_update_NSE[test.sID] + C_update_NSE[test.tID]; 
			
			test.dRatingHat_RMSE += omega_RMSE * ytemp_RMSE;
			test.dRatingHat_MAE += omega_MAE * ytemp_MAE;
			test.dRatingHat_NSE += omega_NSE * ytemp_NSE;
			
			test.dRatingHats_RMSE.add(ytemp_RMSE);
			test.dRatingHats_MAE.add(ytemp_MAE);
			test.dRatingHats_NSE.add(ytemp_NSE);
			
		}
		
		return result_M1_omega;		
	}			
}
