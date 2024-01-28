
import java.io.IOException;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.StringTokenizer;
import java.util.Map;
import java.util.HashMap;
import java.util.Set;
import java.util.HashSet;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class CountWithoutStop {

  public static class TokenizerMapper
       extends Mapper<Object, Text, Text, IntWritable>{

    private final static IntWritable one = new IntWritable(1);
    private Text word = new Text();
    private Set<String> stopWords = new HashSet<>();

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
      // Load stop words from the stop words file
      Configuration conf = context.getConfiguration();
      String exclstpwrds = conf.get("stopwords.file");
      if (exclstpwrds != null && !exclstpwrds.isEmpty()) {
        try (BufferedReader reader = new BufferedReader(new FileReader(exclstpwrds))) {
          String line;
          while ((line = reader.readLine()) != null) {
            stopWords.add(line.trim());
          }
        }
      }
    }

    public void map(Object key, Text value, Context context
                    ) throws IOException, InterruptedException {
      StringTokenizer itr = new StringTokenizer(value.toString());
      while (itr.hasMoreTokens()) {
        String token = itr.nextToken().replaceAll("[^a-zA-Z]","").toLowerCase();  // Remove punctuation and make it lowercase
        if (!token.isEmpty() && !stopWords.contains(token)) {
          word.set(token);
          context.write(word, one);
        }
      }
    }
  }

  public static class IntSumReducer
       extends Reducer<Text, IntWritable, Text, IntWritable> {

    public void reduce(Text key, Iterable<IntWritable> values,
                       Context context) throws IOException, InterruptedException {
      int sum = 0;
      for (IntWritable val : values) {
        sum += val.get();
      }
      context.write(key, new IntWritable(sum));
    }
  }

  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    // Pass the path to the stop words file as a configuration parameter
    conf.set("stopwords.file", "/mnt/c/users/tarun/desktop/dic/hw2/gist_stopwords.txt");
    Job job = Job.getInstance(conf, "word count");
    job.setJarByClass(CountWithoutStop.class);
    job.setMapperClass(TokenizerMapper.class);
    job.setReducerClass(IntSumReducer.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(IntWritable.class);
    FileInputFormat.addInputPath(job, new Path(args[0]));
    FileOutputFormat.setOutputPath(job, new Path(args[1]));
    System.exit(job.waitForCompletion(true) ? 0 : 1);
  }
}
