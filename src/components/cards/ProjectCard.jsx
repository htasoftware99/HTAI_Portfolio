import React from "react";
import styled from "styled-components";

const Card = styled.div`
  width: 330px;
  height: 520px;
  background-color: ${({ theme }) => theme.card};
  cursor: pointer;
  border-radius: 10px;
  box-shadow: 0 0 12px 4px rgba(0, 0, 0, 0.4);
  overflow: hidden;
  padding: 26px 20px;
  display: flex;
  flex-direction: column;
  gap: 14px;
  transition: all 0.5s ease-in-out;
  &:hover {
    transform: translateY(-10px);
    box-shadow: 0 0 50px 4px rgba(0, 0, 0, 0.6);
    filter: brightness(1.1);
  }
`;
const Image = styled.img`
  width: 100%;
  height: 180px;
  flex: 0 0 180px;
  object-fit: cover;
  background-color: ${({ theme }) => theme.white};
  border-radius: 10px;
  box-shadow: 0 0 16px 2px rgba(0, 0, 0, 0.3);
`;
const Tags = styled.div`
  display: inline-flex;
  align-self: flex-start;
  color: ${({ theme }) => theme.primary};
  background: ${({ theme }) => theme.primary + 15};
  border: 1px solid ${({ theme }) => theme.primary + 35};
  border-radius: 8px;
  padding: 5px 10px;
  font-size: 12px;
  font-weight: 600;
  text-transform: capitalize;
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
`;
const Details = styled.div`
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 0px 2px;
  min-height: 0;
  overflow: hidden;
  flex: 1 1 auto;
`;
const Title = styled.div`
  font-size: 20px;
  font-weight: 600;
  color: ${({ theme }) => theme.text_secondary};
  display: -webkit-box;
  max-width: 100%;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  text-overflow: ellipsis;
  flex: 0 0 auto;
`;
const Date = styled.div`
  font-size: 12px;
  margin-left: 2px;
  font-weight: 400;
  color: ${({ theme }) => theme.text_secondary + 80};
  @media only screen and (max-width: 768px) {
    font-size: 10px;
  }
`;
const Description = styled.div`
  position: relative;
  font-weight: 400;
  font-size: 14px;
  line-height: 1.55;
  color: ${({ theme }) => theme.text_secondary + 99};
  overflow: hidden;
  margin-top: 4px;
  display: block;
  max-width: 100%;
  min-height: 50px;   
  max-height: 80px;
  padding-bottom: 0;
  flex-shrink: 0;     

  &::after {
    content: "";
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    height: 60px;
    pointer-events: none;
    background: linear-gradient(
      to bottom,
      transparent 0%,
      ${({ theme }) => theme.card} 100%
    );
  }
`;
const Members = styled.div`
  display: flex;
  align-items: center;
  padding-left: 10px;
  min-height: 38px;
`;
const Avatar = styled.img`
  width: 38px;
  height: 38px;
  border-radius: 50%;
  margin-left: -10px;
  background-color: ${({ theme }) => theme.white};
  box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);
  border: 3px solid ${({ theme }) => theme.card};
`;
const Button = styled.button`
  width: 100%;
  border: 1px solid ${({ theme }) => theme.primary + 70};
  border-radius: 8px;
  background: ${({ theme }) => theme.primary + 12};
  color: ${({ theme }) => theme.primary};
  cursor: pointer;
  padding: 10px 14px;
  font-weight: 600;
  text-align: center;
  transition: all 0.25s ease;
  flex: 0 0 auto;
  margin-top: auto;

  &:hover {
    background: ${({ theme }) => theme.primary + 25};
    transform: translateY(-2px);
  }
`;

const ProjectCard = ({ project, onOpen }) => {
  return (
    <Card>
      <Image src={project.image} />
      <Tags>{project.category}</Tags>
      <Details>
        <Title>{project.title}</Title>
        <Date>{project.date}</Date>
        <Description>{project.description}</Description>
      </Details>
      <Members>
        {project.member?.map((member) => (
          <Avatar key={member.github || member.name} src={member.img} alt={member.name} />
        ))}
      </Members>
      <Button type="button" onClick={() => onOpen(project)}>
        View Details
      </Button>
    </Card>
  );
};

export default ProjectCard;
