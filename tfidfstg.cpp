#include<stdio.h>
#include<iostream>
#include<map>
#include<iostream>
#include<fstream>
#include<algorithm>
#include <mpi.h>
#include<dirent.h>
#include<string.h>
#include<string>
#include<cmath>
#include<stdlib.h>
#include<vector>
#include<omp.h>
#include "CudaCosineSimilarity.hpp"
#include <iomanip>
#define WMAX 1000
#define FMAX 3
#define pb push_back
#define mp make_pair
using namespace std;

//========================================================//
struct dirent* ent;
DIR* books;

typedef struct msg{
	int start,end;
}MSG;
typedef struct movies{
	string name;
	string description;
}mov;
//=========================================================//
bool myFunc(pair<string, float> a,pair<string, float>b){
	return a.second > b.second;
}
//========================================================//
bool sortFunc(pair<int,double>a, pair<int,double>b){
	return a.second < b.second;
}
//========================================================//









std::vector<std::vector<double> > cudaCosineSimilarity(std::vector<std::vector<double> > v1, std::vector<std::vector<double> > v2, double *ms) {
    std::cout << "Running CUDA Cosine Similarity" << std::endl;

    std::clock_t start = std::clock();
    double *results = cudaCosine(v1, v2);
    std::clock_t end = std::clock();

    std::vector<std::vector<double> > t_results;
    int vector_size = v1[0].size();

    std::vector<double> v1_results;
    for(int i = 0; i < v1.size(); i++) {
        v1_results.clear();
        for(int j = 0; j < v2.size(); j++) {
            v1_results.push_back(results[i*v2.size() + j]);
        }
        t_results.push_back(v1_results);
    }

    *ms = (end - start) / (double) (CLOCKS_PER_SEC / 1000);

    return t_results;

}




int main(int argc , char *argv[]){
	double startT, endT;
	const int MASTER  =  0;
	int numTasks,rank,count=0,datawait,rc;
	
	int totalrec;
//	cout<<"Enter Records: ";
//	cin>>totalrec;
//	cout<<endl;

	fstream fptr;
	fptr.open("movies_metadata - Sheet1.csv", ios::in);
	vector<mov> movs; 
	int itera=0;
	while(!fptr.eof() && itera<1001){
		mov m12;
		    
        getline(fptr, m12.name ,',');
        getline(fptr, m12.description ,'\n');
	movs.push_back(m12);
	
	count++;
	//cout<<"\n name: "<<movs[itera].name<<" || description: "<<movs[itera].description<<endl;
	itera++;
	}
	fptr.close();	

	//cout<<"\n\n\n\n\nname: "<<movs[14].name<<end<<endl;
	
	
	
	
	MPI_Status Stat;
	MPI_Request req;
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&numTasks);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);	
	omp_set_num_threads(4);
	/* create a custom datatype */
	const int nitems=2;
	map<string,float>IDF;
	map<string,float>IDF2;
	map<string,float>::iterator it;
	int blocklengths[2] = {1,1};
	MPI_Datatype types[2] = {MPI_INT, MPI_INT};
	MPI_Datatype mpi_msg_type;
	MPI_Aint     offsets[2];

	offsets[0] = offsetof(MSG, start);
	offsets[1] = offsetof(MSG, end);

	MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_msg_type);
	MPI_Type_commit(&mpi_msg_type);
	
	/* MASTER CODE */
	MSG outMsg;
	MSG inMsg;

		//count=movs.size();

	if(rank == MASTER){
		startT = omp_get_wtime();
		
		if(numTasks > count){
			printf("More Number of processes then files in corpus\n");
			exit(0);
		}

		int range = (int)floor(count / (float)(count,numTasks-1));
		printf("\n\n\n\n\nRange: %d\n\n\n\n",range);
		int s = 1,e;
		/* send out Messages to the slaves */
		for(int i = 1 ; i< numTasks ; i++){
			e = s + range - 1-1;
			outMsg.start = s-1;
			outMsg.end = e;
			if(i == numTasks - 1)
				outMsg.end = max(e,count-1);
			s = s+range;
			rc = MPI_Send(&outMsg, 1, mpi_msg_type, i, 1, MPI_COMM_WORLD);
	      	rc = MPI_Send(&count, 1, mpi_msg_type, i, 2, MPI_COMM_WORLD);	
				  printf("Task %d: Sent message %d %dto task %d with tag 1\n",rank, outMsg.start, outMsg.end, i);
		}
		
		// wait for completion message from all idfs. // 
		for(int i = 1 ;  i< numTasks ; i++){
			rc = MPI_Recv(&inMsg, 1, mpi_msg_type, i, 1, MPI_COMM_WORLD, &Stat);
		}
		//printf("IDF re\n");
		
		// compute Global IDF //
		IDF.clear();
		
		float val;
		int cnt;
		char fname[80],temp[8],token[800];
		int flag = 0;
		#pragma omp parallel for private(fname,token,val,cnt)
		for(int i= 1 ; i< numTasks ; i++){
			char fname[80],temp[8],token[800];
			sprintf(fname,"Output/IDF/idf%d.txt",i);
			FILE* fp = fopen(fname,"r");
			int cnt;
			while(fscanf(fp,"%s%s%d",token,temp,&cnt) != EOF ){
				string str(token);
				
				//#pragma omp critical  
				 if(IDF.find(str) == IDF.end()){
					#pragma omp critical
					IDF.insert(mp(str,cnt));
				 }
				 else{
					#pragma omp critical
					IDF[str] += val;
				 }
				
				
			}	
		}
		cout<<"\nidf read successfully\n";
		ofstream idout;
		idout.open("Output/IDF/Corpus_Idf.txt");
		for(it = IDF.begin() ; it != IDF.end() ; it++){
			it -> second = log((count) / it->second);
			
			idout << it -> first << " -> " << it -> second<<endl;
			IDF2.insert(mp(it->first, it->second));
		}
		IDF.clear();
		cout<<"\n corpus idf written successfully\n";


		
		vector<vector<double>> vq;
		float cn12;
		vector<float> tfidf_vectors[count];
		//vector<float> temp_vec(20);
		
		//map<string,float> tfsets[count];
	//	#pragma omp parallel for private(fname,token,val,cn12)
		for(int i= 0 ; i< count ; i++){
			int cnt;
			char fname[80],temp[8],token[800];
			//cout<<endl;
			sprintf(fname,"Output/TF/tf%d.txt",i);
			FILE* fp = fopen(fname,"r");
			//cout<<endl<<endl<<i<<endl<<endl;
			//string word,temp;
			float cn123;
			vector<float> temp_vec(20,0);
			
			 int checkdigi=0;
			 while(fscanf(fp,"%s%f",token,&cn12)!=EOF&&checkdigi<20) {
				
				string str(token);
					
				//tfsets[i].insert(mp(str,cnt));
				if(IDF2.find(str)!=IDF2.end()){
			//	cout <<"tfid value of "<<str<<" is: "<<IDF2[str]*cn12<<"       ";
				
				float valflot=IDF2[str]*cn12;
				//tfidf_vectors[i].push_back(valflot);
				if(checkdigi<20){
				temp_vec[checkdigi]=valflot;
				}
				else{
				//temp_vec.push_back(valflot);
				}
				}

				checkdigi++;

			 }
			vector<double> temp_vec2(temp_vec.begin(),temp_vec.end());
			vq.push_back(temp_vec2);	
			//cout<<endl;
			temp_vec.clear();
		}
		cout<<"\n tfidf calculated successfully\n";
		
		// for(int i=0;i<count;i++){
		// 	vq.push_back(tfidf_vectors[i]);
				
		// }
		endT = omp_get_wtime();
		printf("Timetaken for TFIDF Calculation: %lf\n",endT-startT);

		// for(int i = 1 ;  i< numTasks ; i++){
		// 	rc = MPI_Send(&outMsg, 1, mpi_msg_type, i, 1, MPI_COMM_WORLD);
		// }
		// printf("sent\n");
		// // wait for completion message from all processes. // 
		// for(int i = 1 ;  i< numTasks ; i++){
		// 	rc = MPI_Recv(&inMsg, 1, mpi_msg_type, i, 1, MPI_COMM_WORLD, &Stat);
		// }

		double cudasec;
		vector<vector<double>> results=cudaCosineSimilarity(vq,vq,&cudasec);
		for(int i =0;i<count;i++){
			for(int j=0;j<results[i].size();j++){
			//	cout<<" "<<results[i][j]<<" ";
			}
			//cout<<endl;
		}












		
			
	}
	else{
		double t1,t2;
		t1 = omp_get_wtime();
		map<string, float>TF;
		int count;
		map<string, float>IDF;
		map<string, float>::iterator it;
		map<string, float>::iterator itF;
		IDF.clear();
		int cflag = 0;
		float val;
		string StopWords[400];
		char fname[80],token[800],outFile[80],temp[30];
		memset(fname,'\0',sizeof(fname));
		memset(outFile,'\0',sizeof(outFile));
		/* Read Stop Words From File */
		ifstream stop("StopWords.txt");
		int k = 0;
		while(getline(stop, StopWords[k])){
			k++;
		}
		
		
		rc = MPI_Recv(&inMsg, 1, mpi_msg_type, MASTER, 1, MPI_COMM_WORLD, &Stat);
		rc= MPI_Recv(&count, 1, mpi_msg_type, MASTER, 2, MPI_COMM_WORLD, &Stat);
		printf("Task %d: Received %d char(s) (%d %d) from task %d with tag %d \n",
		rank, count, inMsg.start, inMsg.end,Stat.MPI_SOURCE, Stat.MPI_TAG);
		int idfflag = 0;
	//	map<string, int>tfset[inMsg.end-inMsg.start+1];
	//	int totalsetssthg=inMsg.end- inMsg.start+1;
		int mycont=0;
		#pragma omp parallel for private(fname,outFile,TF,token,it,idfflag) shared(StopWords)
		for(int i = inMsg.start ; i<= inMsg.end; i++){
			idfflag = 0;
			memset(fname,'\0', sizeof(fname));
			memset(outFile,'\0', sizeof(outFile));
			TF.clear();
			
		//	sprintf(fname,"BOOKS/%d.txt",i);
			sprintf(outFile,"Output/TF/tf%d.txt",i);
			// FILE* fp = fopen(fname,"r");
			ofstream out;
			out.open(outFile);
			// if(fp == NULL){
			// 	printf("Directory not open\n");
			// 	exit(0);
			// }
			istringstream isis(movs[i].description);
			int mergeFlag = 0;
			string final_token;
			char sp[2] = {' ','\0'};
			int totwo=0;
			while(isis>>token){
			totwo++;				
				string tok(token);
				tok.erase(remove(tok.begin(), tok.end(), '#'), tok.end());
				tok.erase(remove(tok.begin(), tok.end(), ' '), tok.end());
				tok.erase(remove(tok.begin(), tok.end(), '>'), tok.end());
				tok.erase(remove(tok.begin(), tok.end(), '|'), tok.end());
				tok.erase(remove(tok.begin(), tok.end(), '='), tok.end());
				tok.erase(remove(tok.begin(), tok.end(), '+'), tok.end());
				tok.erase(remove(tok.begin(), tok.end(), '_'), tok.end());
				tok.erase(remove(tok.begin(), tok.end(), ','), tok.end());
				tok.erase(remove(tok.begin(), tok.end(), '-'), tok.end());
				tok.erase(remove(tok.begin(), tok.end(), '*'), tok.end());
				tok.erase(remove(tok.begin(), tok.end(), ';'), tok.end());
				tok.erase(remove(tok.begin(), tok.end(), '"'), tok.end());
				tok.erase(remove(tok.begin(), tok.end(), '.'), tok.end());
				tok.erase(remove(tok.begin(), tok.end(), '!'), tok.end());
				tok.erase(remove(tok.begin(), tok.end(), ')'), tok.end());
				tok.erase(remove(tok.begin(), tok.end(), '('), tok.end());
				tok.erase(remove(tok.begin(), tok.end(), ']'), tok.end());
				tok.erase(remove(tok.begin(), tok.end(), '['), tok.end());
				tok.erase(remove(tok.begin(), tok.end(), '&'), tok.end());
				tok.erase(remove(tok.begin(), tok.end(), '$'), tok.end());
				tok.erase(remove(tok.begin(), tok.end(), '?'), tok.end());
				//transform(tok.begin(), tok.end(), tok.begin(), ::tolower);
				
				int tokFlag= 0;
				string tt = tok;
				transform(tt.begin(), tt.end(), tt.begin(), ::tolower);
				for(int k = 0 ; k < 400 ; k++){
					if(tt.compare(StopWords[k]) == 0){
						tokFlag = 1;
						break;
					}
				}
				if(tokFlag == 1){
					continue;
				}
				
				if(tok[0]>= 65 && tok[0] <= 90 && mergeFlag == 0){
					final_token.clear();
					final_token = tok;
					mergeFlag = 1;
					continue;
				}
				else if(tok[0]>= 65 && tok[0] <= 90 && mergeFlag == 1 && final_token.size() < 100){
					final_token = final_token + "_" +tok;
					continue;
				}
				else if((tok[0]<65 || tok[0] >90) && mergeFlag == 1){
					tok.clear();
					tok = final_token;
					final_token.clear();
					mergeFlag = 0;
				}
				
				if((int)tok[0] == 39 || ((int)tok[0]>=48 && (int)tok[0]<=57)|| tok.size() == 0)
					continue;
				if(TF.find(tok) == TF.end()){
					TF.insert(pair<string, float>(tok,1));
				//tfset[mycont].insert(pair<string, int>(tok,1));
				}
				else{
					TF[tok] +=1;
					//tfset[mycont][tok]+=1;
				}
			
			}
			//out<<movs[i].name<<endl;
			for(it = TF.begin() ; it!= TF.end() ; it++){

				out << it->first<<" "<<( it->second/totwo) << endl;
				if(IDF.find(it->first) == IDF.end()){
					#pragma omp critical
					IDF.insert(pair<string , float>(it->first, 1));
				}
				else{
					#pragma omp critical
					IDF[it ->first] +=1;
				} 

			}
			//tfset[mycont]=TF;
			out.close();
			//fclose(fp);
			mycont++;
			
		}
		cout<<"\nTF calculate by process: "<<rank<<endl;
		ofstream out_idf;
		sprintf(fname,"Output/IDF/idf%d.txt",rank);
		out_idf.open(fname);
		for(itF = IDF.begin() ; itF!= IDF.end() ; itF++){
			//itF -> second = log10((inMsg.end - inMsg.start + 1)/itF->second);
			out_idf << itF->first << " -> "<< itF->second << endl; 
		}
		IDF.clear();
		out_idf.close();
		MSG inM;
		rc = MPI_Send(&inM, 1, mpi_msg_type, MASTER, 1, MPI_COMM_WORLD);
		printf("Local Idf Completed Message sent to Master\n");
		//rc = MPI_Recv(&inM, 1, mpi_msg_type, MASTER, 1, MPI_COMM_WORLD, &Stat);





		// completion Message
		t2 = omp_get_wtime();
		//rc = MPI_Send(&outMsg, 1, mpi_msg_type, MASTER, 1, MPI_COMM_WORLD);
		printf("Time Taken by Process %d is %lf\n",rank,t2-t1);
	}
	MPI_Finalize();
	


return 0;	
}
	