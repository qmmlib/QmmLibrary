#include "do_predict_iter.h"
#include "errno.h"

double do_predict_iter(char* file_name, const struct model* model_)
{
	FILE *input = fopen(file_name,"r");
	int max_nr_attr = 64;
	struct feature_node *x = (struct feature_node *) malloc(max_nr_attr*sizeof(struct feature_node));
	
	int correct = 0;
	int total = 0;
	double error = 0;
	double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;
	
	
	int nr_class=get_nr_class(model_);
	int j, n;
	int nr_feature=get_nr_feature(model_);
	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;

	int max_line_len = 1024;
	char * line = (char *)malloc(max_line_len*sizeof(char));
	int len;
	while(fgets(line,max_line_len,input) != NULL)
	{
		while(strrchr(line,'\n') == NULL)
		{
			max_line_len *= 2;
			line = (char *) realloc(line,max_line_len);
			len = (int) strlen(line);
			if(fgets(line+len,max_line_len-len,input) == NULL)
				break;
		}
	
		int i = 0;
		double target_label, predict_label;
		char *idx, *val, *label, *endptr;
		int inst_max_index = 0; // strtol gives 0 if wrong format

		label = strtok(line," \t\n");
		if(label == NULL) // empty line
		{
			fprintf(stderr,"Wrong input format at line %d\n", total+1);
			exit(1);
		}

		target_label = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
		{
			fprintf(stderr,"Wrong input format at line %d\n", total+1);
			exit(1);
		}

		while(1)
		{
			if(i>=max_nr_attr-2)	// need one more for index = -1
			{
				max_nr_attr *= 2;
				x = (struct feature_node *) realloc(x,max_nr_attr*sizeof(struct feature_node));
			}

			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;
			errno = 0;
			x[i].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x[i].index <= inst_max_index)
			{
				fprintf(stderr,"Wrong input format at line %d\n", total+1);
				exit(1);
			}
			else
				inst_max_index = x[i].index;

			errno = 0;
			x[i].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
			{
				fprintf(stderr,"Wrong input format at line %d\n", total+1);
				exit(1);
			}

			// feature indices larger than those in training are not used
			if(x[i].index <= nr_feature)
				++i;
		}

		if(model_->bias>=0)
		{
			x[i].index = n;
			x[i].value = model_->bias;
			i++;
		}
		x[i].index = -1;

		predict_label = predict(model_,x);

		if(predict_label == target_label)
			++correct;
		error += (predict_label-target_label)*(predict_label-target_label);
                sump += predict_label;
                sumt += target_label;
                sumpp += predict_label*predict_label;
                sumtt += target_label*target_label;
                sumpt += predict_label*target_label;
                ++total;
	}
	free(line);
	free(x);
	fclose(input);
	
	return ((double)correct/(double)total)*100.0f;
}
