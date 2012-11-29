#include "transform_line.h"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

unsigned int split(const std::string &txt, std::vector<std::string> &strs, char ch)
{
    size_t pos = txt.find( ch );
    size_t initialPos = 0;
    strs.clear();

    // Decompose statement
    while( pos != std::string::npos ) {
        strs.push_back( txt.substr( initialPos, pos - initialPos + 1 ) );
        initialPos = pos + 1;

        pos = txt.find( ch, initialPos );
    }

    // Add the last one
    strs.push_back( txt.substr( initialPos, std::min( pos, txt.size() ) - initialPos + 1 ) );

    return strs.size();
}

void split(char* _str, double* read_line, int * y)
{
    int i = 0;
	while(1)
	{
		if(_str[i]==0)
			break;
		else if(_str[i]==':' || _str[i]=='\r' || _str[i]=='\n' || _str[i]=='\t')
			_str[i] = ' ';
		i++;
	}
	
	std::vector<std::string> v;
	std::string str(_str);
	split(str, v, ' ' ); 
	
	bool labeled = false;
	int last_index = 0;
	for(int i=0; i<v.size(); i++)
	{
		string s = v[i];
		if(s.find_first_not_of(' ') != std::string::npos)
		{
			if(!labeled)
			{
				*y = atoi(s.c_str());
				labeled = true;
				continue;
			}
			if(last_index==0)
				last_index = (int) atof(s.c_str());
			else
			{
				read_line[last_index] = atof(s.c_str());
				last_index = 0;
			}
		}
	}
}

void transform_line(double * read_line, int y, int max_idx, char * line, char structured_w, double ** tmx, int tmx_r, int tmx_c)
{
	int curr_idx=1;
	int print_idx=1;
	double curr_val;
	
	stringstream ss (stringstream::in | stringstream::out);

	ss<<y<<" ";
	
	while(1)
	{
		if(structured_w=='s' || structured_w=='a')
		{
			curr_val = read_line[curr_idx] + (read_line[max_idx+1-curr_idx] * (structured_w=='s'?1:(structured_w=='a'?-1:0)));
			
			if(curr_val!=0)
				ss<< print_idx<<":"<<curr_val<<" ";
			
			curr_idx++;
			print_idx++;
			
			if(max_idx%2==0 && curr_idx>((double)max_idx)/2.0f)
				break;
			else if(max_idx%2!=0 && curr_idx>((double)max_idx)/2.0f)
			{
				curr_val = read_line[curr_idx];
				if(curr_val!=0)
					ss<< print_idx<<":"<<curr_val<<" ";
				break;
			}
		}			
		else if(structured_w=='f')
		{
			if(max_idx!=tmx_c)
			{
				printf("\nmax_idx: %d ncols: %d nrows: %d\n",max_idx,tmx_c,tmx_r);
				exit(1);
			}
			
			for(int j = 0; j < tmx_r; j++)
			{
				curr_val = 0;
				for(int i = 0; i<tmx_c; i++)
					curr_val = curr_val + (tmx[j][i] * read_line[i+1]);
				if(curr_val!=0)
					ss<<j+1<<":"<<curr_val<<" ";
			}
			
			break;
		}
	}
	
	strcpy(line, ss.str().c_str());
	// printf("%s\n",line);
}
