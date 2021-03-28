---
layout: page
title: CME 213 Spring 2021 Syllabus
---

This is a 3-unit class. This course is offered online with mixed synchronous/asynchronous engagement only.

This class will give hands-on experience with programming multicore processors, graphics processing units (GPU), and parallel computers. The focus will be on the message passing interface (MPI, parallel clusters) and the compute unified device architecture (CUDA, GPU). Topics will include multithreaded programs, GPU computing, computer cluster programming, C++ threads, OpenMP, CUDA, and MPI. Pre-requisites include C++, templates, debugging, UNIX, makefile, numerical algorithms (differential equations, linear algebra).

## Challenging times 

Students and instructors are all adjusting to the changes and regulations that have been put in place in response to COVID-19. I hope that we will all work together as a community to adapt to these times and this new situation as best as we can. I have tried to add ﬂexibility to our course structure and assignments to accommodate anyone who may now be lacking resources that are normally available. However, if you have suggestions for how we might support your learning in this course during these trying times, please do not hesitate to let us know.

## Survey 

Please go to [Canvas](https://canvas.stanford.edu/courses/133903) to fill the online survey.

## Teaching staff

Instructor: Prof. Eric Darve, [darve@stanford.edu](mailto:darve@stanford.edu). Prof. Darve is a Professor in Mechanical Engineering and a faculty member affiliated with ICME.

Teaching assistants:

- Chenzhuo Zhu, head TA
- Vikul Gupta

How to contact us? The best way to contact us directly is through Slack using Direct messages.

## Lectures and class material

Most of the material for this class will be accessible from the [class website](https://ericdarve.github.io/cme213-spring-2021/). Meeting information (zoom link) and grades are posted on [Canvas](https://canvas.stanford.edu/courses/133903).

Most of the lectures will be pre-recorded. The material will be listed on the class website. It is organized in modules, with videos, slides, and reading and homework assignments.

At this time, we are not planning on many live lectures. There are practical difficulties including network bandwidth, connection reliability, and time difference.

To supplement the lectures, we will have a combination of reading and homework assignments. **Reading assignments** are short questions that can be answered by reading the slides and watching the videos. **Homework assignments** will be based primarily on Python and will require some code development and analysis.

The [class website](https://ericdarve.github.io/cme213-spring-2021/) contains useful books, articles, web links, recorded lectures, etc, that are relevant to this class.

## Forum

To communicate, we will post messages using [canvas announcements](https://canvas.stanford.edu/courses/133903/announcements). There is also a **Slack** workspace. You will need to join the workspace for this class.

1. Go to [https://stanford.enterprise.slack.com/](https://stanford.enterprise.slack.com/)
2. Search for `cme213-spring-2021`.
3. Sign in. You should be able to join immediately.

Check the channels in the workspace. An important channel is `#homework`. The Slack URL is: [https://cme213-spring-2021.slack.com/](https://cme213-spring-2021.slack.com/).

Since the class is taught virtually, it is important to maintain contact with the teaching staff and other students. We encourage you to freely share information, questions, feedback, comments, etc, on Slack in the appropriate channel. Slack is meant to be a flexible and open-ended way to communicate.

It is critical that you send feedback about the class to the teaching staff. It could be some comments, things you wish were done differently, or maybe some special difficulty you are facing right now. For feedback, you can use 

- email Eric Darve [darve@stanford.edu](mailto:darve@stanford.edu)
- send a direct message using Slack to Eric Darve, Chenzhuo Zhu, or Vikul Gupta
- office hours
- use the anonymous Google form (see [Canvas](https://canvas.stanford.edu/courses/133903) for the link). Note that we don't get notified when someone fills the form, so it may take time before we see your message. In addition, since this is anonymous, we won't be able to reply to you.

**Rules of conduct.** On the forum, please observe the following code of conduct:

- Be civil, considerate, and courteous to everyone. The forum is meant to be a safe and welcoming space to get help. If your message is not useful to other students, yourself, or the instructors, you should probably just delete it.
- Access to the various forums will be revoked without warning if you post an inappropriate, disrespectful, demeaning, or abusive message.

## Office hours

The teaching staff will have office hours. These will be posted on canvas. Look under the [Zoom](https://canvas.stanford.edu/courses/133903/external_tools/5384) tab.

During the zoom office hours, we will use a combination of "Waiting Room," which requires the host to let you in, and "Breakout Rooms," which allow splitting participants into small groups (in this case, we will have each student in their own private "room"). This will allow managing one-on-one discussions with potentially more than one participant in zoom.

## Grading

The grading will be done as follows:

- Final project: 35%
- Reading assignments: 25%
- Homework: 40%

We will use [gradescope](https://www.gradescope.com/courses/258024) for homework submission and grading. Search for `CME 213 Spring 2021`. You should be automatically enrolled if you are listed on canvas. The code to enroll (if required) is listed on [Canvas](https://canvas.stanford.edu/courses/133903).

All homework for this class will be prepared electronically. Homework papers will consist of:

- Computer code. We will use primarily C++ and CUDA.
- Written report with answers, plots, and figures in PDF format.

After receiving your grade on gradescope, you are welcome to request a regrade using the gradescope interface. No one is perfect. We strive to grade accurately, fairly, and provide useful feedback to help you, but mistakes do happen. We will be happy to address any concerns you have. However, to help with the logistics, we prefer that you submit your regrade request at most 1 week after the grade has been released.

## Final Project

You will be given a final project to work on towards the end of the quarter. The project will be on implementing a neural network to recognize hand-written digits ([MNIST](http://yann.lecun.com/exdb/mnist/) Modified National Institute of Standards and Technology database). The project will involve CUDA and MPI programming.

## What this class is about

We will focus on how to program:

- Multicore processors, e.g., desktop processors: C++ threads, Pthreads, OpenMP.
- NVIDIA graphics processors using CUDA.
- Computer clusters using MPI.

We will cover some numerical algorithms for illustration: sort, linear algebra, and basic parallel primitives.

## What this class is not about

- Parallel computer architecture
- Parallel design patterns and programming models
- Parallel numerical algorithms. See [CME 342: Parallel Methods in Numerical Analysis](https://explorecourses.stanford.edu/search?view=catalog&filter-coursestatus-Active=on&page=0&catalog=&academicYear=&q=cme342&collapse=).

## Requirements and pre-requisites

- Basic knowledge of UNIX (ssh, compilers, makefile, git)
- Knowledge of C and C++ (including pointers, memory, templates, polymorphism, standard library)
- Proficiency in scientific programming, including **testing, verification, and debugging**

## Computer access

For this class, we will have access to a computer cluster owned by ICME. The computer name is `icme-gpu.stanford.edu`. You will receive instructions on how to use this computer later on. This computer has GPUs which will be required for the GPU homework assignments and the final project. For the first few assignments, you can use your own computer or one of the computers on [FarmShare](https://uit.stanford.edu/service/sharedcomputing/environments); `rice.stanford.edu` is a good option.

## Books

Good news! Most books are available electronically from the Stanford Library. Just go to [http://searchworks.stanford.edu/](http://searchworks.stanford.edu/).

### Parallel computing books

- _Parallel Programming for Multicore and Cluster Systems,_ Rauber and R&uuml;nger. Applications focus mostly on linear algebra.
- _Introduction to Parallel Computing,_ Grama, Gupta, Karypis, Kumar. Wide range of applications from sort to FFT, linear algebra and tree search.
- _An Introduction to Parallel Programming,_ Pacheco. More examples and less theoretical.
- _C++ High Performance: Boost and Optimize the Performance of your C++17 Code,_ Andrist, Sehr; focused on recent C++ features but also contains a discussion of parallel computing for shared memory machines
- _Introduction to High Performance Computing for Scientists and Engineers,_ Hager, Wellein; MPI and openMP; discussion of hybrid parallel computing

### OpenMP and multicore books

- _Using OpenMP: Portable Shared Memory Parallel Programming,_ Chapman, Jost, van der Pas. Advanced coverage of OpenMP.
- _Parallel Programming in OpenMP,_ Chandra, Menon, Dagum, Kohr, Maydan, McDonald; a bit outdated now.
- _The Art of Multiprocessor Programming,_  Herlihy, Shavit. Specializes on advanced multicore programming.
- _Using OpenMP&mdash;The Next Step: Affinity, Accelerators, Tasking, and SIMD,_ van der Pas, Stotzer, Terbo; covers recent extensions to OpenMP and some advanced usage

### CUDA books

- Best reference: _[CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)_ on the NVIDIA web site. This reference is the most up-to-date and comprehensive.
- _Professional CUDA C Programming,_ Cheng, Grossman, McKercher; has more advanced usage like multi-GPU programming
- _Programming Massively Parallel Processors: A Hands-on Approach,_ Kirk, Hwu; in its 3rd edition now; covers a wide range of topics including several in numerical linear algebra, many applications, and parallel programming patterns relevant to CUDA
- _CUDA for Engineers: An Introduction to High-Performance Parallel Computing,_ Storti, Yurtoglu; interesting examples relevant to engineers; relatively short book; good introduction
- _CUDA by Example: An Introduction to General-Purpose GPU Programming,_ Sanders, Kandrot; good introduction; a little outdated now
- _CUDA Handbook: A Comprehensive Guide to GPU Programming,_ Wilt; lots of advanced technical details on memory, streaming, the CUDA compiler, examples of CUDA optimizations

### MPI books

- _Parallel Programming with MPI,_ Pacheco; classic reference; somewhat dated at this point
- _Using MPI: Portable Parallel Programming with the Message-Passing Interface,_ Gropp, Lusk, Skjellum; very complete reference
- _Using Advanced MPI: Modern Features of the Message-Passing Interface,_ Gropp, Hoefler, Thakur, Lusk; same authors as previous entry; discusses recent and more advanced features of MPI

## What you can expect from me 

I am here to guide your learning and will challenge you to actively engage in the learning process through class activities, assignments, and more. I will strive for an inclusive and collaborative classroom and welcome any suggestions for improvement. I will do my best to give you the tools, feedback, and support to succeed, so let me know if I can do anything more. Learning is a never-ending process, so I hope to motivate students to seek out more information on topics we don't have time to cover. I highly encourage everyone to visit me in office hours or to set up a meeting, even if you don't feel that you have questions. I want to get to know you and support you in this learning experience! The best way to reach me is by email/Slack (see contact information) and you can expect me to respond within 24 hours (Monday-Friday).

## What I expect from you 

It can be easy to get distracted in a virtual learning environment and during online meetings. So, I ask that you try as best as you can to remain focused and engaged in the class material. I expect you to be proactive and take an active role in your learning by following attentively our lectures and being ready to collaborate with your classmates. Moreover, online settings can often feel anonymous and less personal, sometimes making it easier to misinterpret comments or to share thoughts with less filtering. Keep in mind that each member of this class has different ideas and perspectives that will enrich the experience for us all. I expect all of us to speak and listen with compassion and not make assumptions about others. Never hesitate to email me, join me in my virtual office hours, or set up a meeting. This class should challenge you, but I believe everyone has the ability to succeed with some effort.

## Respect for diversity

It is my intent that students from all diverse backgrounds, perspectives, and situations be well served by this course, that students' learning needs be addressed both in and out of class, and that the diversity that students bring to this class be viewed as a resource, strength and benefit. It is my intent to present materials and activities that are respectful of diversity, which may include but not limited to: gender, sexuality, disability, age, socioeconomic status, ethnicity, race, religion, political affiliation, culture, and so on. I acknowledge that there is likely to be a diversity of access to resources among students and plan to support all of you as best as I can. Please let me know ways to improve the effectiveness of the course for you personally or for other students or student groups. In addition, if any of our class meetings conflict with your religious events, please let me know so that we can make arrangements for you.

All people have the right to be addressed and referred to in accordance with their personal identity. In this class, we will have the chance to indicate the name that we prefer to be called  (see the [Survey](https://docs.google.com/forms/d/e/1FAIpQLSdxd1rdTGfqHA7eGehUoLmVjeXHruqoR4IjYOeOE8cTdSa2wA/viewform?usp=sf_link)) and, if we choose, to identify pronouns with which we would like to be addressed. I will do my best to address and refer to all students accordingly and support classmates in doing so as well.

[Diversity-related resources](https://diversityworks.stanford.edu/resource#select=.activities-and-support.students) at Stanford

## Support services

The COVID-19 pandemic is a stressful time for us all. In addition, you may experience a range of other challenges that can cause barriers to learning, such as strained relationships, increased anxiety, alcohol/drug problems, feeling down, difficulty concentrating and/or lack of motivation. These mental health concerns or stressful events may lead to diminished academic performance or reduce your ability to participate in daily life. Stanford is committed to advancing the mental health and well-being of its students. If you or someone you know is feeling overwhelmed, depressed, and/or in need of support, services are available. 

To learn more about the broad range of confidential mental health services available on campus, please see the list of [resources on mental health](https://vaden.stanford.edu/health-resources/mental-health) at Stanford.

[Vaden Counseling & Psychological Services](https://vaden.stanford.edu/caps)

## Students with Documented Disabilities

Students who may need an academic accommodation based on the impact of a disability must initiate the request with the Office of Accessible Education (OAE). Professional staff will evaluate the request, review appropriate medical documentation, recommend reasonable accommodations, and prepare an Accommodation Letter for faculty dated in the current quarter in which the request is being made. The letter will indicate how long it is to be in effect. Students should contact the OAE as soon as possible since timely notice is needed to coordinate accommodations. 

The OAE is located at 563 Salvatierra Walk.

Phone: (650) 723-1066; email: [oae-contactus@stanford.edu](mailto:oae-contactus@stanford.edu)

Office Hours: Monday&ndash;Friday, 9 AM&ndash;5 PM

URL: [https://oae.stanford.edu](https://oae.stanford.edu)

<br/>

**We hope you will enjoy this class and find it useful!**

<br/>

## Honor Code and Office of Community Standards

We take the honor code very seriously. The honor code is Stanford's statement on academic integrity first written by Stanford students in 1921. It articulates university expectations of students and faculty in establishing and maintaining the highest standards in academic work. It is agreed to by every student who enrolls and by every instructor who accepts appointment at Stanford. The Honor Code states:

1. The Honor Code is an undertaking of the students, individually and collectively

    (a) that they will not give or receive aid in examinations; that they will not give or receive unpermitted aid in class work, in the preparation of reports, or in any other work that is to be used by the instructor as the basis of grading;

    (b) that they will do their share and take an active part in seeing to it that others as well as themselves uphold the spirit and letter of the Honor Code.

2. The faculty on its part manifests its confidence in the honor of its students by refraining from proctoring examinations and from taking unusual and unreasonable precautions to prevent the forms of dishonesty mentioned above. The faculty will also avoid, as far as practicable, academic procedures that create temptations to violate the Honor Code.
3. While the faculty alone has the right and obligation to set academic requirements, the students and faculty will work together to establish optimal conditions for honorable academic work.

Note that the student who lets others copy his work is as guilty as those who copy. Violations include at least the following circumstances: copying material from another student, copying previous year solution sets, copying solutions found using Google, copying solutions found on the internet. You will be automatically reported without a warning if a violation is suspected. The Office of Community Standards is in charge of determining whether a violation actually occurred or not.

Please do not post any material from this class online. This will encourage honor code violation, and penalize other students. This is also a violation of copyright.

If found guilty of a violation, your grade will be automatically lowered by at least one letter grade, and the instructor may decide to give you a "No Pass" or "No Credit" grade. The standard sanction from OCS for a first offense includes a one-quarter suspension from the University and 40 hours of community service. For multiple violations (e.g., cheating more than once in the same course), the standard sanction is a three-quarter suspension and 40 or more hours of community service.

[Honor Code statement and information](https://communitystandards.stanford.edu/policies-and-guidance/honor-code)