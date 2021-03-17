use std::env;
use std::fs;
use std::cmp::{max, min};
use std::collections::HashMap;


fn main() {
    // Read input from command line
    let vals: Vec<String> = env::args().collect();

    let argc = vals.len();
    if argc < 3 || argc > 3 {
        println!("ERROR: Need three arguments exactly.");
        panic!("Malformed input");
    }
   
    let gt_filepath = &vals[1];
    let dets_filepath = &vals[2];
    println!("Reading ground truth file from: {}", gt_filepath);
    println!("Reading predictions file from: {}", dets_filepath);
    
    // Read the annotations from each file
    let gt_annotations = parse_annotation_file(gt_filepath);
    let det_annotations = parse_annotation_file(dets_filepath);

    let iou_threshold = 0.5;
    let AP = compute_AP(&gt_annotations, &det_annotations, iou_threshold);

}

#[derive(Debug, Clone)]
struct Annotation {
    frame: usize,
    class: usize,
    x_min: f32,
    y_min: f32,
    width: f32,
    height: f32 
}

impl Annotation {
    fn from_vec(values: Vec<&str>) -> Annotation {
        if values.len() < 8 {
            println!("Values: {:?}", values);
            panic!("Not enough values");
        }

        Annotation {
            frame: values[0].parse().unwrap(),
            class: 0,
            x_min: values[2].parse().unwrap(),
            y_min: values[3].parse().unwrap(),
            width: values[4].parse().unwrap(),
            height: values[5].parse().unwrap() 
        }
    }
}

// Returns a Hashmap of vectors of annotations
fn parse_annotation_file(file_path_str: &String) -> HashMap<usize, Vec<Annotation>> { 
    // Read the file into a string
    let annotations_str = fs::read_to_string(file_path_str)
        .expect("Something went wrong reading the file");

    // Parse the string into a vector of annotations
    let mut annotations: HashMap<usize, Vec<Annotation>> = HashMap::new();
    for annotation_str in annotations_str.split("\n") { 
        if annotation_str.len() < 8 {
            continue;
        }

        else {
            let temp: Vec<&str> = annotation_str.split(",").collect();
            let ann = Annotation::from_vec(temp);

            if annotations.contains_key(&ann.frame) {
                annotations.get_mut(&ann.frame).unwrap().push(ann.clone());
            }

            else {
                let mut annotation_vec: Vec<Annotation> = Vec::new();
                annotation_vec.push(ann.clone());
                annotations.insert(ann.frame, annotation_vec);
            }
        }
    }
 
    // Return the vector of annotations
    annotations
}


fn compute_AP(gt_annotations: &HashMap<usize, Vec<Annotation>>, dets: &HashMap<usize, Vec<Annotation>>, iou_threshold:f32) -> f64 { 
    
    // For each frame, find the detections that meet the IOU threshold with the gts (ground truths)
    for (frame, predictions) in dets.iter() {
         
        if let Some(gts) = gt_annotations.get(&frame) {
            let matches = find_matches(&gts, &predictions);

        } 
    }

    0.0
}


fn find_matches(gt_annotations: &Vec<Annotation>, dets: &Vec<Annotation>) {
    for gt in gt_annotations.iter(){
        for det in dets.iter() {
            let iou_val = iou(gt, det);
            println!("{:?}", iou_val);

        }
    }
}


fn iou(gt: &Annotation, pred: &Annotation) -> f64 {
    
    let gt_max_x = gt.x_min + gt.width;
    let gt_max_y = gt.y_min + gt.height;
    let pred_max_x = pred.x_min + pred.width;
    let pred_max_y = pred.y_min + pred.height;

    // determine the (x, y)-coordinates of the intersection rectangle
    let x_a = 0.0;
    println!("{:?}", x_a);
    // # determine the (x, y)-coordinates of the intersection rectangle
    // xA = max(boxA[0], boxB[0])
    // yA = max(boxA[1], boxB[1])
    // xB = min(boxA[2], boxB[2])
    // yB = min(boxA[3], boxB[3])
    // # compute the area of intersection rectangle
    // interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    // # compute the area of both the prediction and ground-truth
    // # rectangles
    // boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    // boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    // # compute the intersection over union by taking the intersection
    // # area and dividing it by the sum of prediction + ground-truth
    // # areas - the interesection area
    // iou = interArea / float(boxAArea + boxBArea - interArea)


    0.0
}
